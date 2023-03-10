
# -*- coding: utf-8 -*-
import os
import kuramoto
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from scipy import stats, signal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


dt, t_step, T = 0.1, 10, 150000

seed = 42
np.random.seed(seed)


def plt_pause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


class Kuramoto(kuramoto.Kuramoto):

    def frequencies(self, act_mat, adj_mat):
        assert len(adj_mat) == act_mat.shape[0], 'adj_mat does not match act_mat'
        _, n_steps = act_mat.shape

        n_interactions = (adj_mat != 0).sum(axis=0)  # number of incoming interactions
        coupling = self.coupling / n_interactions  # normalize coupling by number of interactions

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for time in range(n_steps):
            dxdt[:, time] = self.derivative(act_mat[:, time], None, adj_mat, coupling=coupling)
        return dxdt

    def mean_frequency(self, act_mat, adj_mat):
        """
        Compute average frequency within the time window (self.T) for all nodes
        """

        dxdt = self.frequencies(act_mat, adj_mat)

        # Integrate all nodes over the time window T
        integral = np.sum(dxdt * self.dt, axis=1)
        # Average across complete time window - mean angular velocity (freq.)
        meanfreq = integral / self.T
        return meanfreq


def run(_graph: np.ndarray, _angles_vec_init: np.ndarray, _nat_freqs_init: np.ndarray):

    model = Kuramoto(coupling=1,
                     dt=dt,
                     T=t_step,
                     n_nodes=len(_graph),
                     natfreqs=_nat_freqs_init,
                     )

    activity_ = model.run(
        adj_mat=_graph,
        angles_vec=_angles_vec_init,
    )

    frequencies_ = model.frequencies(act_mat=activity_, adj_mat=_graph)

    return activity_, frequencies_


if __name__ == '__main__':
    n_neurons = 6

    graph_nx = nx.erdos_renyi_graph(n=n_neurons, p=1)     # ????????????????????????

    # graph_nx = nx.circulant_graph(n_neurons, [1])   # ????????????

    # graph_nx = nx.Graph()   # ????????????
    # graph_nx.add_nodes_from(range(n_neurons))
    # graph_nx.add_edges_from(zip([0] * (n_neurons - 1), range(1, n_neurons)))

    # graph_nx = nx.Graph()  # ??????????????
    # graph_nx.add_nodes_from(range(n_neurons))
    # graph_nx.add_edges_from(zip(range(0, n_neurons - 1), range(1, n_neurons)))

    # graph_nx = nx.erdos_renyi_graph(n=n_neurons, p=0.5)     # ??????????????????

    pos = graphviz_layout(graph_nx, prog='dot')
    nx.draw(graph_nx, pos, node_color='lightblue', edge_color='#909090', node_size=200, with_labels=True)
    plt.show()

    graph = nx.to_numpy_array(graph_nx) * np.random.rand(n_neurons, n_neurons) * .4
    # graph = nx.to_numpy_array(graph_nx)

    angles_vec = 2 * np.pi * np.random.random(size=n_neurons) - np.pi
    # natfreqs = np.random.normal(loc=1., scale=0.1, size=n_neurons)
    natfreqs = np.random.normal(size=n_neurons)

    plt.ion()
    fig, axes = plt.subplots(figsize=(10, 7), ncols=1, nrows=3)
    axes[0].set_ylabel(r'$\sin(\theta)$', rotation=0)
    axes[1].set_ylabel('Coherence')
    axes[2].set_ylabel(r'$\omega$', rotation=0)
    plt.show(block=False)

    x_time = np.array([])
    y_angles = np.array([[]] * n_neurons)
    y_phase_coherence = np.array([])
    x_time2 = [0]
    y_mean_freqs = [natfreqs]
    # x_time3 = []
    # y_partial_stat = []

    for j, t_curr in enumerate(np.arange(0., T + t_step, t_step)):
        activity, frequencies = run(_graph=graph, _angles_vec_init=angles_vec, _nat_freqs_init=natfreqs)

        phase_coherence = np.asarray([Kuramoto.phase_coherence(vec) for vec in activity.T])
        angles_vec = activity.T[-1]

        freqs_vec = frequencies.T[-1]
        mean_freqs = np.sum(frequencies * dt, axis=1) / t_step

        natfreqs[:] = mean_freqs[:]
        # natfreqs[:] = freqs_vec[:]

        x_time = np.hstack((x_time, np.arange(t_curr, t_curr + t_step, dt)))
        y_angles = np.hstack((y_angles, np.sin(activity)))
        y_phase_coherence = np.hstack((y_phase_coherence, phase_coherence))
        x_time2.append(t_curr + t_step)
        y_mean_freqs.append(mean_freqs)

        # plt.gca().clear()
        for i in range(3):
            axes[i].clear()

        axes[0].set_ylabel(r'$\sin(\theta)$', rotation=0)
        axes[1].set_ylabel('Coherence')
        axes[2].set_ylabel(r'$\overline{\omega} = $' + '{:.3f}'.format(np.mean(mean_freqs)), rotation=0)

        N = int(t_step/dt)

        axes[0].plot(x_time[-10*N:], y_angles.T[-10*N:])
        axes[1].plot(x_time[-100*N:], y_phase_coherence[-100*N:], color='forestgreen')
        axes[2].plot(x_time2[-100:], y_mean_freqs[-100:])

        # if not os.path.exists('anim'):
        #     os.makedirs('anim')
        # plt.savefig(os.path.join('anim', '{}.png'.format(str(j).zfill(8))), dpi=50)

        if np.all(np.isclose(phase_coherence, 1.)):
            print('\n???????????? ??????????????????????????!')
            break

        # # if t_curr % (N * 10) == 0 and j != 0:
        # if t_curr > N * 10:
        #     prev = np.asarray([y_phase_coherence[k:k + len(phase_coherence)]
        #                        for k in range(-N * 10, -len(phase_coherence) * 2, 1)])
        #     corr = np.asarray([stats.pearsonr(phase_coherence, p) for p in prev])
        #
        #     x_time3.append(t_curr)
        #     y_partial_stat.append([np.max(corr), np.mean(corr), np.min(corr)])
        #     # print('max: {:.4f} \t mean: {:.4f} \t min: {:.4f}'.format(np.max(corr), np.mean(corr), np.min(corr)))
        #     axes[3].plot(x_time3[-100:], y_partial_stat[-100:])
        #
        #     if np.any(np.isclose(corr, 1.)):
        #         print('\n?????????????????? ??????????????????????????!')
        #         break

        plt_pause(0.00000000001)

    plt.show(block=True)
