#!/usr/bin/env python
from __future__ import division
# import itertools
import os

import numpy as np
# from scipy.optimize import minimize
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

q_var = 4
q_std = np.sqrt(q_var)

RES_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_ops(m):
    if '_log_sum_rat' not in m.ops:
        with m.sess.graph.as_default():
            s = m.test_data
            logp = m.ops['fv']
            logq = (-.5 * m.target.D * np.log(2 * np.pi * q_var)
                    - tf.reduce_sum(s ** 2, axis=1) / (2 * q_var))
            m.ops['_log_rat'] = diff = logp - logq
            m.ops['_log_sum_rat'] = tf.reduce_logsumexp(diff)
            # XXX are we worried about catastrophic cancellation type stuff here?
            # maybe we should be....
            m.ops['_log_sum_rat_sq'] = tf.reduce_logsumexp(2 * diff)


def run_batch_q0(m, ops, n, batch_size=10**6, pbar=None):
    n_batches = int(np.ceil(float(n) / batch_size))
    sizes = np.full(n_batches, batch_size)
    sizes[:n - sizes.sum()] += 1
    assert sizes.sum() == n

    D = m.target.D
    return np.array([
        m.sess.run(ops, feed_dict={
            m.test_data: np.random.normal(scale=q_std, size=(sz, D))})
        for sz in (sizes if pbar is None else pbar(sizes))
    ])


def est_mean(m, n, include_var=False, **batch_kwargs):
    ensure_ops(m)
    ops = [m.ops['_log_sum_rat']]
    if include_var:
        ops.append(m.ops['_log_sum_rat_sq'])

    bits = run_batch_q0(m, ops, n, **batch_kwargs)
    log_means = logsumexp(bits, axis=0) - np.log(n)
    log_mean_est = log_means[0]

    if include_var:
        log_mean_sq_est = log_means[1]
        log_var_est = (
            log_mean_sq_est
            + np.log1p(-np.exp(2 * log_mean_est - log_mean_sq_est))
            + np.log(n) - np.log(n - 1)  # silly Bessel correction
        )
        return log_mean_est, log_var_est
    else:
        return log_mean_est


def est_log_percentile(m, p, n, **batch_kwargs):
    # do mean of percentile on each batch, for memory efficiency + simplicity
    # reasonable if batch_size is decently large...
    ensure_ops(m)
    with m.sess.graph.as_default():
        log_rat = m.ops['_log_rat']
        pct = tf.contrib.distributions.percentile(log_rat, float(p))
    ests, = run_batch_q0(m, [pct], n, **batch_kwargs)
    return logsumexp(ests) - np.log(ests.shape[0])


def est_cdf(m, x, n, **batch_kwargs):
    ensure_ops(m)
    with m.sess.graph.as_default():
        diff = m.ops['_log_rat']
        le = tf.count_nonzero(diff <= x)
    ests, = run_batch_q0(m, [le], n, **batch_kwargs)
    return ests.sum() / n


def estimate_bias(model, n_pct=10**6, n_hoeffding=10**7, hoeffding_delta=.001,
                  n_var=10**7, n_psi=10**7, **batch_kw):
    D = model.target.D

    # find the probability-1 lower bound, a
    alpha = model.sess.run(model.alpha)
    log_a = alpha[alpha < 0].sum() + .5 * D * np.log(2 * np.pi * q_var)

    # find the point s at the 40th percentile
    if 'batch_size' in batch_kw:
        assert batch_kw['batch_size'] >= 10**4
    log_s = est_log_percentile(model, 40, n_pct, **batch_kw)

    # get Hoeffding bound on cdf at s:
    #   Pr(\sum 1(X <= x) - E[1(X <= x)] > eps) <= exp(- 2 n eps^2) = delta
    #   eps = sqrt(log(1/delta) / (2n))
    p_hat = est_cdf(model, log_s, n_hoeffding, **batch_kw)
    rho = p_hat + np.sqrt(-np.log(hoeffding_delta) / (2 * n_hoeffding))
    assert rho < .5

    # estimate the variance
    log_Z_for_var, log_var_1 = est_mean(
        model, n_var, include_var=True, **batch_kw)

    # estimate of the normalizer to use in the bound
    log_Z = est_mean(model, n_var, **batch_kw)

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


def compute_for(dset, seed, gpu_count=0, cpu_count=None, **kwargs):
    pth = os.path.join(RES_DIR, 'bias/{}/{}.npz'.format(dset, seed))
    if os.path.exists(pth):
        return dict(**np.load(pth))

    if not os.path.exists(os.path.dirname(pth)):
        try:
            os.makedirs(os.path.dirname(pth))
        except OSError as e:
            if e.errno == 17:
                pass
            else:
                raise
    np.savez(pth, in_progress=True)

    p = load_data(dset, seed=seed, itanh=False, whiten=True)
    model = DeepLite(p, seed=seed, gpu_count=gpu_count, cpu_count=cpu_count,
                     **dl_args)
    model.load()

    res = estimate_bias(model, **kwargs)
    np.savez(pth, **res)
    return res


def bias_est(n_samps, bias_info):
    return (
        np.exp(bias_info['log_term1'] - np.log(n_samps))
        + bias_info['scale'] * np.exp(n_samps * bias_info['log_base']))


def bias_est_for(dset, seed, n_samps, **kwargs):
    return bias_est(n_samps, compute_for(dset, seed, **kwargs))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsets', nargs='+')
    parser.add_argument('--seeds', nargs='+', default=range(15), type=int)
    parser.add_argument('--n-psi', type=int, default=10**7)
    parser.add_argument('--n-var', type=int, default=10**7)
    parser.add_argument('--n-pct', type=int, default=10**6)
    parser.add_argument('--n-hoeffding', type=int, default=10**6)
    parser.add_argument('--hoeffding-delta', type=int, default=.001)
    parser.add_argument('--batch-size', type=int, default=10**6)

    parser.add_argument('--gpu-count', default=0, type=int)
    parser.add_argument('--cpu-count', default=None, type=int)
    args = parser.parse_args()

    kwargs = vars(args).copy()
    del kwargs['dsets'], kwargs['seeds']

    maybe_tqdm = lambda x, **k: tqdm(x, **k) if len(x) > 1 else x
    for dset in maybe_tqdm(args.dsets):
        tqdm.write("Starting on {}".format(dset))
        for seed in maybe_tqdm(args.seeds):
            res = compute_for(dset, seed, **kwargs)
            if 'in_progress' in res:
                tqdm.write("WARNING: {}/{} says it's in progress; did it crash?"
                           .format(dset, seed))


if __name__ == '__main__':
    main()
