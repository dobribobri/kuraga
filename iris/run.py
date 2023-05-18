# -*- coding: utf-8 -*-
import dill
import tqdm
import numpy as np
from kkuramoto import Kuramoto
from multiprocessing import Pool
from stats import Stats


n_workers = 8
K_range = np.linspace(0.001, 1, 240)

dt, t_step, T = 0.1, 10, 15000

seed = 42
np.random.seed(seed)

adj_mat = np.load('adjmat.npy')
omega = np.load('omega.npy')


def run(K: float) -> tuple[float, np.ndarray, int]:
    N = len(adj_mat)
    stats = Stats(N)  # 150

    phase_vec_init = np.zeros(N)
    nat_freqs_init = omega

    t_curr = 0.

    phase_vec_curr = phase_vec_init
    freqs_vec_curr = nat_freqs_init
    while t_curr <= T:
        model = Kuramoto(coupling=1, dt=dt, T=t_step, n_nodes=N, natfreqs=freqs_vec_curr, )
        activity = model.run(adj_mat=adj_mat * K, angles_vec=phase_vec_curr)
        frequencies = model.frequencies(act_mat=activity, adj_mat=adj_mat * K)
        phase_coherence = np.asarray([Kuramoto.phase_coherence(vec) for vec in activity.T])

        phase_vec_curr = activity.T[-1]
        last_freqs = frequencies.T[-1]

        mean_freqs = np.sum(frequencies * dt, axis=1) / t_step

        freqs_vec_curr = last_freqs

        stats.update(_time=np.linspace(t_curr, t_curr + t_step, len(activity.T)),
                     _phases=np.sin(activity), _frequencies=frequencies, _phase_coherence=phase_coherence,
                     _time_=t_curr + t_step, _last_freqs=last_freqs, _mean_freqs=mean_freqs)

        t_curr += t_step

        if np.all(np.isclose(phase_coherence, 1.)):  # Полная синхронизация
            return K, mean_freqs, 1

        if len(stats.MEAN_FREQS) > 20:  # Частичная синхронизация!
            cr1 = np.linalg.norm(np.asarray(stats.MEAN_FREQS[-20:]) - mean_freqs)
            stats.F_NORM.append(cr1)
            if len(stats.F_NORM) > 20:
                f_norm = np.asarray(stats.F_NORM[-20:])
                cr2 = np.linalg.norm(f_norm - np.mean(f_norm))
                if cr1 < 0.1 or cr2 < 0.1:
                    return K, mean_freqs, 2

    return K, np.array([None] * len(adj_mat)), 0


if __name__ == '__main__':
    results = []

    with Pool(processes=n_workers) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(run, K_range), total=len(K_range)):
            results.append(result)

    results = sorted(results, key=lambda item: item[0])

    with open('results.bin', 'wb') as dump:
        dill.dump(results, dump)
