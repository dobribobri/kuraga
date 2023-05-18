# -*- coding: utf-8 -*-
import sys
import itertools
from collections import defaultdict
import numpy as np
import networkx as nx
import kuramoto
from sklearn import datasets, preprocessing, model_selection, metrics
from scipy.special import softmax
from multiprocessing import Manager, Process, Pool
import tqdm
from matplotlib import pyplot as plt


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


class OscillatoryNeuralNetwork:
    def __init__(self, input_layer_size: int, output_layer_size: int,
                 hidden_layer_sizes=(4,),
                 oriented: bool = False,
                 coupling_global: float = 10.,
                 dt: float = 0.1, t_step: float = 10, T: float = 100):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = []
        self.n_nodes = input_layer_size + int(np.sum(hidden_layer_sizes)) + output_layer_size
        self.sizes = [input_layer_size] + [ls for ls in hidden_layer_sizes if ls] + [output_layer_size]
        self.G = OscillatoryNeuralNetwork.__multilayered_graph(oriented, *self.sizes)
        self.coupling_global = coupling_global
        self.adj_mat = nx.to_numpy_array(self.G) * self.coupling_global
        self.adj_mat *= np.random.random(self.adj_mat.shape)
        self.adj_mat += 0.1
        self.input_layer_size, self.output_layer_size = input_layer_size, output_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dt, self.t_step, self.T = dt, t_step, T
        self.genes = self.genes = list(itertools.product(range(self.n_nodes), range(self.n_nodes)))

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

    def run(self, _angles_vec_init: np.ndarray, _nat_freqs_init: np.ndarray,):
        time_ = np.array([])
        angles_ = np.array([[]] * self.n_nodes)
        frequencies_ = np.array([[]] * self.n_nodes)
        phase_coherence_ = np.array([])
        angles_vec = _angles_vec_init
        natfreqs = _nat_freqs_init
        for t_curr in np.arange(0., self.T, self.t_step):
            model = Kuramoto(coupling=1, dt=self.dt, T=self.t_step, n_nodes=self.n_nodes, natfreqs=natfreqs,)
            activity = model.run(adj_mat=self.adj_mat, angles_vec=angles_vec)
            frequencies = model.frequencies(act_mat=activity, adj_mat=self.adj_mat)
            phase_coherence = np.asarray([Kuramoto.phase_coherence(vec) for vec in activity.T])
            time_ = np.hstack((time_, np.arange(t_curr, t_curr + self.t_step, self.dt)))
            angles_ = np.hstack((angles_, np.sin(activity)))
            frequencies_ = np.hstack((frequencies_, frequencies))
            phase_coherence_ = np.hstack((phase_coherence_, phase_coherence))
            if np.all(np.isclose(phase_coherence, 1.)):
                break
            angles_vec = activity.T[-1]
            natfreqs = frequencies.T[-1]
        return time_, angles_, frequencies_, phase_coherence_

    def forward(self, _x: np.ndarray, _angles_vec_init: np.ndarray, _apply=lambda x: x) -> np.ndarray:
        assert self.input_layer_size == len(_x), 'wrong shape for input data'
        _, _, frequencies, _ = self.run(_angles_vec_init=_angles_vec_init,
                                        _nat_freqs_init=np.concatenate((_x, np.zeros(self.n_nodes - len(_x)))))
        out = frequencies.T[-1][-self.output_layer_size:]
        return _apply(out)

    def forward_multiple(self, _X: np.ndarray, _angles_vec_init: np.ndarray, _apply=lambda x: x) -> np.ndarray:
        out = np.asarray([self.forward(_x, _angles_vec_init) for _x in _X])
        return _apply(out)

    def error_on_batch(self, _X: np.ndarray, _y_true: np.ndarray, _angles_vec_init: np.ndarray) -> float:
        assert _X.shape[0] == _y_true.shape[0], 'X and y_true must have the same number of samples'
        assert self.output_layer_size == _y_true.shape[1], 'wrong shape (y_true)'
        out = self.forward_multiple(_X, _angles_vec_init, _apply=lambda b: softmax(b, axis=1))
        return np.linalg.norm(_y_true - out)

    def crossover(self, obj: 'OscillatoryNeuralNetwork', percentage: float = 0.5) -> None:
        sh, sw = self.adj_mat.shape
        oh, ow = obj.adj_mat.shape
        assert sh == sw == oh == ow == self.n_nodes, 'wrong shape'
        transfer = [obj.genes[_i] for _i in np.random.choice(range(len(obj.genes)), int(len(obj.genes) * percentage))]
        for _i, _j in transfer:
            self.adj_mat[_i][_j] = obj.adj_mat[_i][_j]

    def mutation(self, n: int):
        mutable = [self.genes[_i] for _i in np.random.choice(range(len(self.genes)), n)]
        for _i, _j in mutable:
            self.adj_mat[_i][_j] = np.random.random() * self.coupling_global


if __name__ == '__main__':
    print('Hello!')
    X, y = datasets.load_iris(return_X_y=True)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)
    y_categorical = np.zeros((len(y), len(label_encoder.classes_)))
    for i, j in enumerate(y):
        y_categorical[i][j] = 1
    y = y_categorical
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    n_workers = 8
    n_individuals = 100
    n_best = 10
    n_new = 10
    n_genes_mutable = 10
    n_epochs = sys.maxsize

    initial_population = []
    for i in range(n_individuals):
        initial_population.append(
            OscillatoryNeuralNetwork(input_layer_size=4, output_layer_size=3, hidden_layer_sizes=None)
        )
    population = initial_population

    angles_vec_init = np.zeros(4 + 3)

    plt.ion()
    fig, ax = plt.subplots(nrows=2, figsize=(10, 7))
    # ax[0].set_title('loss')
    # ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('value')
    # ax[1].set_title('f1-score')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel(r'$\%$')

    losses = defaultdict(list)
    f1_train_list, f1_test_list = [], []
    colors = ['crimson', 'forestgreen', 'black']

    def process(args) -> tuple[float, float]:
        _i, _individual = args
        return _i, _individual.error_on_batch(X_train, y_train, angles_vec_init)


    for epoch in range(n_epochs):
        print('Epoch #{}'.format(epoch))

        scores = []
        with Pool(processes=n_workers) as pool:
            for score in tqdm.tqdm(pool.imap_unordered(process, enumerate(population)),
                                   total=len(population)):
                scores.append(score)
            # scores = pool.imap_unordered(process, enumerate(population))

        errors = sorted(scores, key=lambda item: item[1])
        values = [val for _, val in errors]

        s = '\r{:.2f}%\t-\tepoch: {}\t-\tloss: {:.2f} (1 best),\t{:.2f} ({} best),\t{:.2f} (total)\t-\t'.format(
            (epoch + 1.) / n_epochs * 100,
            epoch + 1,
            np.mean(values[:1]),
            np.mean(values[:n_best]),
            n_best,
            np.mean(values)
        )
        losses['best'].append(np.mean(values[:1]))
        losses['{} best'.format(n_best)].append(np.mean(values[:n_best]))
        losses['total'].append(np.mean(values))
        epochs = np.array(list(range(0, epoch + 1))) + 1
        for i, key in enumerate(losses.keys()):
            ax[0].plot(epochs, losses[key], label='loss: {}'.format(key), color=colors[i])
        if not epoch:
            ax[0].legend(loc='best', frameon=False)

        best, other = [i for i, _ in errors[:n_best]], [i for i, _ in errors[n_best:]]
        p = np.asarray(population, dtype=object)
        best_individuals, other_individuals = p[best], p[other]

        best_out_train = best_individuals[0].forward_multiple(X_train, angles_vec_init,
                                                              _apply=lambda _y: softmax(_y, axis=1))
        best_out_test = best_individuals[0].forward_multiple(X_test, angles_vec_init,
                                                             _apply=lambda _y: softmax(_y, axis=1))
        # print(y_train)
        # print(best_out_train)
        f1_train = metrics.f1_score(y_train, best_out_train, average='weighted')
        f1_test = metrics.f1_score(y_test, best_out_test, average='weighted')
        s += 'f1: {:.2f} (train),\t{:.2f} (test)'.format(
            f1_train, f1_test
        )
        print(s, end='', flush=True)

        f1_train_list.append(f1_train * 100.)
        f1_test_list.append(f1_test * 100.)
        ax[1].plot(epochs, f1_train_list, label='f1-score: train', color=colors[0])
        ax[1].plot(epochs, f1_test_list, label='f1-score: test', color=colors[1])
        if not epoch:
            ax[1].legend(loc='best', frameon=False)

        plt.savefig('evolution.png', dpi=300)
        plt.show(block=False)
        fig.canvas.flush_events()

        np.random.shuffle(best_individuals)
        np.random.shuffle(other_individuals)
        # Новая кровь
        for i in range(len(other_individuals) - n_new, len(other_individuals)):
            other_individuals[i] = OscillatoryNeuralNetwork(input_layer_size=4, output_layer_size=3,
                                                            hidden_layer_sizes=None)
        # Скрещивание
        print('{} | crossover'.format(s), end='', flush=True)
        for individual in other_individuals[-n_new:]:
            best_parent = np.random.choice(best_individuals)
            individual.crossover(best_parent, percentage=0.5)

        # Мутация
        print('{} | mutation'.format(s), end='', flush=True)
        for individual in other_individuals[-n_new:]:
            individual.mutation(n=n_genes_mutable)
        print(s, end='', flush=True)
        new_population = best_individuals.tolist() + other_individuals.tolist()
        np.random.shuffle(new_population)
        population = new_population

    plt.show()
    plt.ioff()
    plt.close()
