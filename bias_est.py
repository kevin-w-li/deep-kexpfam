#!/usr/bin/env python
from __future__ import division
import os

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
from scipy.special import logsumexp
import tensorflow as tf
from tqdm import tqdm

from Datasets import load_data
from LiteModels import DeepLite


dl_args = dict(
    npoint=300, nlayer=3, nneuron=30, init_log_lam=-3, points_std=0.0,
    keep_prob=1.0, init_weight_std=1.0, noise_std=0.05,
    points_type="opt", log_lam_weights=-6, step_size=1e-2, mixture_kernel=False,
    init_log_sigma=np.linspace(0, 1, 3), base=True, niter=10000,
    ntrain=100, nvalid=100, patience=200, clip_score=False,
    curve_penalty=True, train_stage=2)


def run_batches(m, op_names, n, q_std=None, le_cutoff=None, batch_size=10**6,
                pbar=None, pbar_args={}, allow_inf=False):
    n_batches = int(np.ceil(float(n) / batch_size))
    sizes = np.full(n_batches, batch_size)
    sizes[-1] = n - (n_batches - 1) * batch_size
    assert sizes.sum() == n
    assert 1 <= sizes[-1] <= batch_size

    fd = {}
    if le_cutoff is not None:
        fd[m.nodes['le_cutoff']] = le_cutoff

    if q_std is None:
        ops = [m.nodes['q_' + name] for name in op_names]
    else:
        ops = [m.nodes[name] for name in op_names]
        fd['q_std'] = q_std

    res = []
    for sz in (pbar(sizes, **pbar_args) if pbar else sizes):
        fd[m.nodes['n_rand']] = sz
        res.append(m.sess.run(ops, feed_dict=fd))
        if not allow_inf:
            assert np.all(np.isfinite(res[-1]))
    return np.asarray(res)


def est_mean(m, n, include_var=False, q_std=None, **batch_kwargs):
    op_names = ['lse_logr', 'lse_2logr'] if include_var else ['lse_logr']
    bits = run_batches(m, op_names, n, q_std=q_std, **batch_kwargs)
    log_means = logsumexp(bits, axis=0) - np.log(n)
    log_mean_est = log_means[0]

    if include_var:
        assert len(log_means) == 2
        log_mean_sq_est = log_means[1]
        log_var_est = (
            log_mean_sq_est
            + np.log1p(-np.exp(2 * log_mean_est - log_mean_sq_est))
            + np.log(n) - np.log(n - 1)  # silly Bessel correction
        )
        return log_mean_est, log_var_est
    else:
        assert len(log_means) == 1
        return log_mean_est


def est_log_percentile(m, p, n, q_std=None, **batch_kwargs):
    log_rats = run_batches(m, ['logr'], n, q_std=q_std, **batch_kwargs)
    log_rats = np.concatenate(np.squeeze(log_rats, 1), 0)
    assert log_rats.shape == (n,)
    assert np.all(np.isfinite(log_rats))
    return np.percentile(log_rats, p)


def est_cdf(m, x, n, q_std=None, **batch_kwargs):
    ests = run_batches(m, ['logr_le'], n, le_cutoff=x,
                       q_std=q_std, **batch_kwargs)
    assert ests.shape[1] == 1
    return ests.sum() / n


def estimate_bias(model, n_pct=10**6, n_hoeffding=10**7,
                  percentile=40, hoeffding_delta=.001,
                  n_var=10**7, n_psi=10**7, **batch_kw):
    # find the probability-1 lower bound, a
    log_a = model.sess.run(model.nodes['q_logr_lowerbound'])

    # find the point s at the 40th percentile
    log_s = est_log_percentile(model, percentile, n_pct,
                               pbar_args=dict(desc="1. percentile"), **batch_kw)

    # get Hoeffding bound on cdf at s:
    #   Pr(\sum 1(X <= x) - E[1(X <= x)] > eps) <= exp(- 2 n eps^2) = delta
    #   eps = sqrt(log(1/delta) / (2n))
    p_hat = est_cdf(model, log_s, n_hoeffding,
                    pbar_args=dict(desc="2. hoeffding "), **batch_kw)
    rho = p_hat + np.sqrt(-np.log(hoeffding_delta) / (2 * n_hoeffding))
    assert rho < .5, "!!!: {}, {}".format(rho, p_hat)

    # estimate the variance
    log_Z_for_var, log_var_1 = est_mean(
        model, n_var, include_var=True,
        pbar_args=dict(desc="3. variance  "), **batch_kw)

    # estimate of the normalizer to use in the bound
    log_Z = est_mean(model, n_var,
                     pbar_args=dict(desc="4. mean      "), **batch_kw)

    # the total bound is:
    # psi(q, Z) = log(Z/q) + q/Z - 1
    # bias <= (
    #     psi(t, Z) / (Z - t)^2 Var[rat] / num_bias
    #   + max(psi(a,Z), psi(t,Z)) (4 * rho * (1-rho))^(num_bias/2)
    #   )
    # where t = (s + a) / 2

    log_t = logsumexp([log_a, log_s]) - np.log(2)

    # Note that psi(a, Z) > psi(t, Z) as long as t < Z
    assert log_t < log_Z

    # first term, except for num_bias:
    t_over_Z = np.exp(log_t - log_Z)
    log_num = np.log(log_Z - log_t + t_over_Z - 1)
    log_den = 2 * (log_Z + np.log1p(-t_over_Z))
    log_term1 = log_num - log_den + log_var_1

    # second term, except for num_bias:
    term2_scale = log_Z - log_a + np.exp(log_a - log_Z) - 1
    term2_log_base = np.log(2) + np.log(rho)/2 + np.log1p(-rho)/2

    # final bound is, if you estimate the normalizer with U iid samples:
    #   np.exp(log_term1 - np.log(U)) + scale * np.exp(U * log_base)
    return {
        'log_a': log_a,
        'log_s': log_s,
        'log_t': log_t,

        'log_term1': log_term1,
        'term2_scale': term2_scale,
        'term2_log_base': term2_log_base,

        'n_pct': n_pct,
        'n_hoeffding': n_hoeffding,
        'hoeffding_delta': hoeffding_delta,
        'p_hat': p_hat,
        'rho': rho,

        'n_var': n_var,
        'log_Z_for_var': log_Z_for_var,
        'log_var_1': log_var_1,

        'n_psi': n_psi,
        'log_Z_psi': log_Z,
    }


def compute_for(dset, seed, gpu_count=0, cpu_count=None,
                bias_seed_offset=None, load_only=False,
                base_dir=None, **kwargs):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'bias')

    pth = os.path.join(base_dir, '{}/{}.npz'.format(dset, seed))
    if os.path.exists(pth):
        res = dict(**np.load(pth))
        if 'in_progress' in res:
            raise ValueError("in progress")
        return res
    if load_only:
        raise ValueError("not computed yet and asked for load_only")

    if not os.path.exists(os.path.dirname(pth)):
        try:
            os.makedirs(os.path.dirname(pth))
        except OSError as e:
            if e.errno == 17:
                pass
            else:
                raise
    np.savez(pth, in_progress=True)

    model, p = load_model(dset, seed, gpu_count, cpu_count)
    if bias_seed_offset is not None:
        np.random.seed(seed + bias_seed_offset)
        with model.sess.graph.as_default():
            tf.set_random_seed(seed + bias_seed_offset)
    res = estimate_bias(model, **kwargs)
    np.savez(pth, **res)
    return res


def load_model(dset, seed, gpu_count=0, cpu_count=None):
    p = load_data(dset, seed=seed, itanh=False, whiten=True)
    model = DeepLite(p, seed=seed, gpu_count=gpu_count, cpu_count=cpu_count,
                     **dl_args)
    model.load()

    return model, p


def bias_est(n_samps, bias_info):
    return (
        np.exp(bias_info['log_term1'] - np.log(n_samps))
        + bias_info['term2_scale'] * np.exp(
                n_samps * bias_info['term2_log_base']))


def bias_est_for(dset, seed, n_samps, **kwargs):
    return bias_est(n_samps, compute_for(dset, seed, **kwargs))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsets', nargs='+')
    parser.add_argument('--seeds', nargs='+', default=range(15), type=int)
    parser.add_argument('--n-psi', type=int, default=10**8)
    parser.add_argument('--n-var', type=int, default=10**8)
    parser.add_argument('--n-pct', type=int, default=10**7)
    parser.add_argument('--n-hoeffding', type=int, default=10**7)
    parser.add_argument('--hoeffding-delta', type=int, default=.001)
    parser.add_argument('--percentile', type=float, default=40)
    parser.add_argument('--batch-size', type=int, default=10**6)
    parser.add_argument('--bias-seed-offset', type=int, default=17)
    parser.add_argument('--pbar', action='store_true', default=False)

    parser.add_argument('--gpu-count', default=0, type=int)
    parser.add_argument('--cpu-count', default=None, type=int)
    parser.add_argument('--base-dir', type=os.path.abspath)
    args = parser.parse_args()

    kwargs = vars(args).copy()
    del kwargs['dsets'], kwargs['seeds']
    if kwargs['pbar'] is True:
        kwargs['pbar'] = tqdm

    def maybe_tqdm(x, **k):
        return tqdm(x, **k) if len(x) > 1 else x

    for dset in maybe_tqdm(args.dsets):
        tqdm.write("Starting on {}".format(dset))
        for seed in maybe_tqdm(args.seeds):
            try:
                res = compute_for(dset, seed, **kwargs)
                if 'in_progress' in res:
                    raise ValueError()
            except ValueError:
                tqdm.write("WARNING: {}/{} says it's in progress; did it crash?"
                           .format(dset, seed))


if __name__ == '__main__':
    main()
