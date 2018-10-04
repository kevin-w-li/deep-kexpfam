#!/usr/bin/env python
from __future__ import division
import os

import numpy as np
from tqdm import tqdm

from bias_est import est_mean, load_model, RES_DIR


def compute_for(dset, seed, n=10**8, est_seed_offset=None,
                gpu_count=0, cpu_count=None, batch_size=10**6, pbar=None):
    pth = os.path.join(RES_DIR, 'norms/{}/{}.txt'.format(dset, seed))
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

    model, p = load_model(dset, seed, gpu_count, cpu_count)
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
    parser.add_argument('--est-seed-offset', type=int, default=12)

    parser.add_argument('--gpu-count', default=0, type=int)
    parser.add_argument('--cpu-count', default=None, type=int)
    args = parser.parse_args()

    kwargs = vars(args).copy()
    del kwargs['dsets'], kwargs['seeds']

    maybe_tqdm = lambda x, **k: tqdm(x, **k) if len(x) > 1 else x
    for dset in maybe_tqdm(args.dsets):
        tqdm.write("Starting on {}".format(dset))
        for seed in maybe_tqdm(args.seeds):
            res = compute_for(dset, seed, pbar=tqdm, **kwargs)


if __name__ == '__main__':
    main()
