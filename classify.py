
# -*- coding: utf-8 -*-
import itertools
import numpy as np
import networkx as nx
import kuramoto
from sklearn import datasets


seed = 42
np.random.seed(seed)


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


class TimeDomain:
    def __init__(self, dt: float = 0.1, t_step: float = 10, T: float = 150000):
        self.dt = dt
        self.t_step = t_step
        self.T = T

    def get_parameters(self):
        return self.dt, self.t_step, self.T


class OscillatoryNeuralNetwork:
    def __init__(self, input_layer_size: int, output_layer_size: int,
                 hidden_layer_sizes=(4,),
                 oriented: bool = False,
                 coupling_global: float = 1.,):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = []
        self.n_nodes = input_layer_size + int(np.sum(hidden_layer_sizes)) + output_layer_size
        self.sizes = [input_layer_size] + [ls for ls in hidden_layer_sizes if ls] + [output_layer_size]
        self.G = OscillatoryNeuralNetwork.__multilayered_graph(oriented, *self.sizes)
        self.coupling_global = coupling_global
        self.adj_mat = nx.to_numpy_array(self.G) * self.coupling_global
        self.adj_mat *= np.random.random(self.adj_mat.shape)
        self.input_layer_size, self.output_layer_size = input_layer_size, output_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes

    def run(self, time_domain: TimeDomain, angles_vec_init: np.ndarray, nat_freqs_init: np.ndarray,):
        dt, t_step, T = time_domain.get_parameters()
        time_ = np.array([])
        angles_ = np.array([[]] * self.n_nodes)
        frequencies_ = np.array([[]] * self.n_nodes)
        phase_coherence_ = np.array([])
        angles_vec = angles_vec_init
        natfreqs = nat_freqs_init
        for t_curr in np.arange(0., T, t_step):
            model = Kuramoto(coupling=1, dt=dt, T=t_step, n_nodes=self.n_nodes, natfreqs=natfreqs,)
            activity = model.run(adj_mat=self.adj_mat, angles_vec=angles_vec)
            frequencies = model.frequencies(act_mat=activity, adj_mat=self.adj_mat)
            phase_coherence = np.asarray([Kuramoto.phase_coherence(vec) for vec in activity.T])
            time_ = np.hstack((time_, np.arange(t_curr, t_curr + t_step, dt)))
            angles_ = np.hstack((angles_, np.sin(activity)))
            frequencies_ = np.hstack((frequencies_, frequencies))
            phase_coherence_ = np.hstack((phase_coherence_, phase_coherence))
            if np.all(np.isclose(phase_coherence, 1.)):
                break
            angles_vec = activity.T[-1]
            natfreqs = frequencies.T[-1]
        return time_, angles_, frequencies_, phase_coherence_

    def forward_frequencies(self, x: np.ndarray, time_domain: TimeDomain, angles_vec_init: np.ndarray) -> np.ndarray:
        assert self.input_layer_size == len(x), 'wrong shape for input data'
        _, _, frequencies, _ = self.run(time_domain, angles_vec_init, nat_freqs_init=x)
        return frequencies.T[-1][-self.output_layer_size]


    @staticmethod
    def __multilayered_graph(oriented=False, *subset_sizes) -> nx.Graph:
        extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
        layers = [range(start, end) for start, end in extents]
        if oriented:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for _i, layer in enumerate(layers):
            G.add_nodes_from(layer, layer=_i)
        for layer1, layer2 in nx.utils.pairwise(layers):
            G.add_edges_from(itertools.product(layer1, layer2))
        return G



if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    print(X.shape, Y.shape)