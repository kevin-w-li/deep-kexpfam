#!/usr/bin/env python
from __future__ import division
import ast
import os

from bias_est import est_mean, load_model

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def compute_for(dset, seed, n=10**8, est_seed_offset=None, base_dir=None,
                gpu_count=0, cpu_count=None, batch_size=10**6, pbar=None,
                load_args={}):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'norms')

    s = ''.join('-{}_{}'.format(k, v) for k, v in sorted(load_args.items()))
    pth = os.path.join(base_dir, dset, '{}{}.txt'.format(seed, s))
    if os.path.exists(pth):
        log_Z, n_already = np.loadtxt(pth)
        if n_already != n:
            raise ValueError("asked for {:,} but cached was {:,}".format(n, n_already))
        return log_Z

    if not os.path.exists(os.path.dirname(pth)):
        try:
            os.makedirs(os.path.dirname(pth))
        except OSError as e:
            if e.errno == 17:
                pass
            else:
                raise

    model, p = load_model(dset, seed, gpu_count, cpu_count, **load_args)
    if est_seed_offset is not None:
        np.random.seed(seed + est_seed_offset)
    log_Z = est_mean(model, n=n, batch_size=batch_size, pbar=pbar)

    np.savetxt(pth, [log_Z, n])
    return log_Z


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsets', nargs='+')
    parser.add_argument('--seeds', nargs='+', default=range(15), type=int)
    parser.add_argument('-n', type=int, default=10**9)
    parser.add_argument('--batch-size', type=int, default=10**6)
    parser.add_argument('--est-seed-offset', type=int, default=12)

    parser.add_argument('--gpu-count', default=0, type=int)
    parser.add_argument('--cpu-count', default=None, type=int)
    parser.add_argument('--base-dir', type=os.path.abspath)
    parser.add_argument('--load-args', default={}, type=ast.literal_eval)
    args = parser.parse_args()

    kwargs = vars(args).copy()
    del kwargs['dsets'], kwargs['seeds']

    def maybe_tqdm(x, **k):
        return tqdm(x, **k) if len(x) > 1 else x

    for dset in maybe_tqdm(args.dsets):
        tqdm.write("Starting on {}".format(dset))
        for seed in maybe_tqdm(args.seeds):
            try:
                compute_for(dset, seed, pbar=tqdm, **kwargs)
            except ValueError as e:
                tqdm.write("Bad model: {}/{}: {}".format(dset, seed, e))
            except tf.errors.NotFoundError as e:
                tqdm.write("{}/{} not found: {}".format(dset, seed, e))


if __name__ == '__main__':
    main()
