import os
import numpy as np
from maf.util import load
#from DKEFModels import DeepLite
from LiteModels import DeepLite
import h5py as h5
from Datasets import load_data

def create_model_id(data_name, model_name, mode, n_hiddens, act_fun, n_comps, seed=None, patience=200):
    """
    Creates an identifier for the provided model description.
    """

    delim = '_'
    id = data_name + delim + model_name + delim

    if mode is not None:
        if mode == 'sequential':
            id += 'seq' + delim
        elif mode == 'random':
            id += 'rnd' + delim
        else:
            raise ValueError('invalid mode')

    for h in n_hiddens:
        id += str(h) + delim

    if n_comps is not None:
        id += 'layers' + delim + str(n_comps) + delim

    id += act_fun

    id += delim + "p%d" % patience
    if seed is not None:
        id += delim + "s%02d" % seed


    return id


def load_all_models(data_name, seed, dl_args, others_args, skip_theano=False):

    p = load_data(data_name, seed=seed, itanh=False, whiten=True)

    loglik = dict()
    models = dict()
    samples = dict()


    n_hiddens = others_args["n_hiddens"]
    n_comps = others_args["n_comps"]
    n_layers = others_args["n_layers"]
    act_fun = others_args["act_fun"]
    mode =  others_args["mode"]

    dl_model = DeepLite(p, seed=seed, **dl_args)
    dl_model.load()
    models["dkef"] = dl_model

    fn = "results/train_sample/%s.h5" %dl_model.default_file_name()

    if os.path.isfile(fn):
        with h5.File(fn,'r') as f:
            assert np.allclose(f["idx"].value,  p.idx)
            loglik["dkef"] = f["loglik_clean"].value

    if skip_theano:
        return p, models, loglik, samples

    model_fn = create_model_id(data_name, "made", mode, n_hiddens, act_fun, None, seed=seed)
    model_fn = "maf_models/%s/%s.pkl" % (p.name.lower(), model_fn)
    models["made"] = load(model_fn)

    fn = "data/made/%s_D%02d_n%d_nn%d_nt200_p200_made_samples_s%02d.h5" % (p.name, p.D, p.noise_std*100, n_hiddens[0], seed)
    if os.path.isfile(fn):
        with h5.File(fn,'r') as f:
            assert np.allclose(f["idx"].value,  p.idx)
            loglik["made"] = f["loglik_clean"].value
            samples["made"] = f["samples"].value


    model_fn = create_model_id(data_name, "mog_made", mode, n_hiddens, act_fun, n_comps, seed=seed)
    model_fn = "maf_models/%s/%s.pkl" % (p.name.lower(), model_fn)
    models["made_mog"] = load(model_fn)

    fn = "data/mog_made/%s_D%02d_n%d_nn%d_nt200_p200_mog_made_samples_s%02d.h5" % (p.name, p.D, p.noise_std*100, n_hiddens[0], seed)
    if os.path.isfile(fn):
        with h5.File(fn,'r') as f:
            assert np.allclose(f["idx"].value,  p.idx)
            loglik["made_mog"] = f["loglik_clean"].value
            samples["made_mog"] = f["samples"].value


    model_fn = create_model_id(data_name, "realnvp", None, n_hiddens, "tanhrelu", n_layers, seed=seed)
    model_fn = "maf_models/%s/%s.pkl" % (p.name.lower(), model_fn)
    models["nvp"] = load(model_fn)
    fn = "data/nvp/%s_D%02d_n%d_nn%d_nl%d_nt200_p200_nvp_samples_s%02d.h5" % (p.name, p.D, p.noise_std*100, n_hiddens[0], n_layers, seed)
    if os.path.isfile(fn):
        with h5.File(fn,'r') as f:
            assert np.allclose(f["idx"].value,  p.idx)
            loglik["nvp"] = f["loglik_clean"].value
            samples["nvp"] = f["samples"].value


    model_fn = create_model_id(data_name, "maf", mode, n_hiddens, act_fun, n_layers, seed=seed)
    model_fn = "maf_models/%s/%s.pkl" % (p.name.lower(), model_fn)
    models["maf"] = load(model_fn)
    fn = "data/maf/%s_D%02d_n%d_nn%d_nl%d_nt200_p200_maf_samples_s%02d.h5" % (p.name, p.D, p.noise_std*100, n_hiddens[0], n_layers, seed)
    if os.path.isfile(fn):
        with h5.File(fn,'r') as f:
            assert np.allclose(f["idx"].value,  p.idx)
            loglik["maf"] = f["loglik_clean"].value
            samples["maf"] = f["samples"].value


    model_fn = create_model_id(data_name, "mog_maf", mode, n_hiddens, act_fun, [n_layers, n_comps], seed=seed)
    model_fn = "maf_models/%s/%s.pkl" % (p.name.lower(), model_fn)
    models["maf_mog"] = load(model_fn)
    fn = "data/mog_maf/%s_D%02d_n%d_nn%d_nl%d_nt200_p200_mog_maf_samples_s%02d.h5" % (p.name, p.D, p.noise_std*100, n_hiddens[0], n_layers, seed)
    if os.path.isfile(fn):
        with h5.File(fn,'r') as f:
            assert np.allclose(f["idx"].value,  p.idx)
            loglik["maf_mog"] = f["loglik_clean"].value
            samples["maf_mog"] = f["samples"].value

    return p, models, loglik, samples
