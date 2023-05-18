# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from kkuramoto import Kuramoto
from switch import Switch
from enum import Enum
from tkinter.messagebox import showinfo
import threading


class TopologyMode(Enum):
    FULL = 'Полносвязный граф'
    STAR = 'Звезда'
    RING = 'Кольцо'
    LINE = 'Линейка'
    RAND = 'Случайная'


class InitialPhaseMode(Enum):
    ZERO = 'Нулевые'
    RAND = 'Случайные в диапазоне от -pi до pi'


class InitialFrequencyMode(Enum):
    ZERO = 'Нулевые'
    RAND1 = 'Случайные в диапазоне от 0 до 1'
    RAND2 = 'Случайный множитель от -1 до 1'


class AdjMatMultiplierMode(Enum):
    NO = 'Бинарная'
    RAND1 = 'Случайный множитель от 0 до 1'
    RAND2 = 'Случайный множитель от -1 до 1'


class ONN:
    def __init__(self, _dt: float = 0.1, _t_step: float = 10., _T: float = 15000,
                 _N: int = 7, _K: float = 0.2,
                 _p: float = .6,
                 _seed: int = 42,

                 _topology_mode: TopologyMode = TopologyMode.RAND,
                 _adjmat_multiplier_mode: AdjMatMultiplierMode = AdjMatMultiplierMode.RAND1,
                 _initial_phase_mode: InitialPhaseMode = InitialPhaseMode.ZERO,
                 _initial_frequency_mode: InitialFrequencyMode = InitialFrequencyMode.RAND1,):

        self._dt, self._t_step, self.T = _dt, _t_step, _T
        self._N, self._K = _N, _K
        self._p = _p
        self._seed = _seed

        self._topology_mode = _topology_mode
        self._multiplier_mode = _adjmat_multiplier_mode

        self.initial_phase_mode = _initial_phase_mode
        self.initial_frequency_mode = _initial_frequency_mode

        self.graph_nx = self.new_graph_nx()
        self.adj_mat = self.new_adjmat()

        self.phase_vec_init = self.new_phase_vec_init()
        self.nat_freqs_init = self.new_nat_freqs_init()

        self.TIME, self.PHASES, self.FREQUENCIES, self.PHASE_COHERENCE = None, None, None, None
        self.TIME_, self.MEAN_FREQS, self.LAST_FREQS, self.F_NORM = None, None, None, None
        self.clear_stats()

        self.N_lock = threading.Lock()
        self.dt_lock = threading.Lock()
        self.t_step_lock = threading.Lock()

    def clear_stats(self):
        self.TIME = np.array([])
        self.PHASES = np.array([[]] * self._N)
        self.FREQUENCIES = np.array([[]] * self._N)
        self.PHASE_COHERENCE = np.array([])
        self.TIME_ = []
        self.MEAN_FREQS = []
        self.LAST_FREQS = []
        self.F_NORM = []

    def update_stats(self, _time, _phases, _frequencies, _phase_coherence,
                     _time_, _mean_freqs, _last_freqs, _f_norm=None):
        self.TIME = np.hstack((self.TIME, _time))
        self.PHASES = np.hstack((self.PHASES, _phases))
        self.FREQUENCIES = np.hstack((self.FREQUENCIES, _frequencies))
        self.PHASE_COHERENCE = np.hstack((self.PHASE_COHERENCE, _phase_coherence))
        self.TIME_.append(_time_)
        self.MEAN_FREQS.append(_mean_freqs)
        self.LAST_FREQS.append(_last_freqs)
        if _f_norm:
            self.F_NORM.append(_f_norm)

    def new_phase_vec_init(self):
        with Switch(self.initial_phase_mode) as case:
            if case(InitialPhaseMode.ZERO):
                phase_vec_init = np.zeros(self.N)
            if case(InitialPhaseMode.RAND):
                phase_vec_init = 2 * np.pi * np.random.random(size=self.N) - np.pi
            if case.default:
                phase_vec_init = np.zeros(self.N)
        return phase_vec_init

    def new_nat_freqs_init(self):
        with Switch(self.initial_frequency_mode) as case:
            if case(InitialFrequencyMode.ZERO):
                nat_freqs_init = np.zeros(self.N)
            if case(InitialFrequencyMode.RAND1):
                nat_freqs_init = np.random.random(size=self.N)
            if case(InitialFrequencyMode.RAND2):
                nat_freqs_init = 2 * np.random.random(size=self.N) - 1
            if case.default:
                nat_freqs_init = np.zeros(self.N)
        return nat_freqs_init

    def new_graph_nx(self):
        with Switch(self.topology_mode) as case:
            if case(TopologyMode.FULL):
                graph_nx = nx.erdos_renyi_graph(n=self.N, p=1)
            if case(TopologyMode.STAR):
                graph_nx = nx.Graph()   # звезда
                graph_nx.add_nodes_from(range(self.N))
                graph_nx.add_edges_from(zip([0] * (self.N - 1), range(1, self.N)))
            if case(TopologyMode.RING):
                graph_nx = nx.circulant_graph(self.N, [1])
            if case(TopologyMode.LINE):
                graph_nx = nx.Graph()  # линейка
                graph_nx.add_nodes_from(range(self.N))
                graph_nx.add_edges_from(zip(range(0, self.N - 1), range(1, self.N)))
            if case(TopologyMode.RAND):
                graph_nx = nx.erdos_renyi_graph(n=self.N, p=self.p)
            if case.default:
                graph_nx = nx.erdos_renyi_graph(n=self.N, p=1)
        return graph_nx

    def new_adjmat(self) -> np.ndarray:
        adj_mat = nx.to_numpy_array(self.graph_nx) * self.K
        with Switch(self.multiplier_mode) as case:
            if case(AdjMatMultiplierMode.NO):
                pass
            if case(AdjMatMultiplierMode.RAND1):
                adj_mat *= np.random.rand(self.N, self.N)
            if case(AdjMatMultiplierMode.RAND2):
                adj_mat *= (2 * np.random.rand(self.N, self.N) - 1)
            if case.default:
                pass
        return adj_mat

    @property
    def topology_mode(self):
        return self._topology_mode

    @topology_mode.setter
    def topology_mode(self, _topology_mode: TopologyMode):
        self._topology_mode = _topology_mode
        self.graph_nx = self.new_graph_nx()
        adj_mat = self.new_adjmat()
        cond = np.logical_not(np.isclose(adj_mat, 0)) & np.logical_not(np.isclose(self.adj_mat, 0))
        adj_mat[cond] = self.adj_mat[cond]
        self.adj_mat = adj_mat

    @property
    def multiplier_mode(self):
        return self._multiplier_mode

    @multiplier_mode.setter
    def multiplier_mode(self, _multiplier_mode):
        self._multiplier_mode = _multiplier_mode
        self.adj_mat = self.new_adjmat()

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, _K: float):
        self.adj_mat = self.adj_mat / self._K * _K
        self._K = _K

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, _N: int):
        self.N_lock.acquire()
        assert _N > 0, ''
        self._N = _N
        self.graph_nx = self.new_graph_nx()
        N_ = len(self.adj_mat)
        if _N <= N_:
            self.adj_mat = self.adj_mat[:_N, :_N]
            self.phase_vec_init = self.phase_vec_init[:_N]
            self.nat_freqs_init = self.nat_freqs_init[:_N]
        else:
            adj_mat = self.new_adjmat()
            adj_mat[:N_, :N_] = self.adj_mat[:, :]
            self.adj_mat = adj_mat
            phase_vec_init = self.new_phase_vec_init()
            phase_vec_init[:N_] = self.phase_vec_init
            self.phase_vec_init = phase_vec_init
            nat_freqs_init = self.new_nat_freqs_init()
            nat_freqs_init[:N_] = self.nat_freqs_init
            self.nat_freqs_init = nat_freqs_init
        self.clear_stats()
        self.N_lock.release()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, _p: float):
        self._p = _p
        self.__setattr__('topology_mode', self._topology_mode)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, _seed: int):
        self._seed = _seed
        np.random.seed(_seed)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, _dt):
        self.dt_lock.acquire()
        self._dt = _dt
        self.dt_lock.release()

    @property
    def t_step(self):
        return self._t_step

    @t_step.setter
    def t_step(self, _t_step):
        self.t_step_lock.acquire()
        self._t_step = _t_step
        self.t_step_lock.release()

    def run(self, _axes, _canvas, _pause_event, _stop_event, _save_plots):

        self.clear_stats()

        self.phase_vec_init = self.new_phase_vec_init()
        self.nat_freqs_init = self.new_nat_freqs_init()

        t_curr = 0.
        freqs_vec_curr = self.nat_freqs_init
        phase_vec_curr = self.phase_vec_init
        while (t_curr <= self.T) and not(_stop_event.is_set()):

            self.N_lock.acquire()
            if self.N <= len(freqs_vec_curr):
                freqs_vec_curr = freqs_vec_curr[:self.N]
            else:
                nat_freqs_init = self.new_nat_freqs_init()
                nat_freqs_init[:len(freqs_vec_curr)] = freqs_vec_curr
                freqs_vec_curr = nat_freqs_init

            if self.N <= len(phase_vec_curr):
                phase_vec_curr = phase_vec_curr[:self.N]
            else:
                phase_vec_init = self.new_phase_vec_init()
                phase_vec_init[:len(phase_vec_curr)] = phase_vec_curr
                phase_vec_curr = phase_vec_init

            self.dt_lock.acquire()
            self.t_step_lock.acquire()
            ####################################################################################################
            model = Kuramoto(coupling=1, dt=self.dt, T=self.t_step, n_nodes=self.N, natfreqs=freqs_vec_curr, )
            activity = model.run(adj_mat=self.adj_mat, angles_vec=phase_vec_curr)
            frequencies = model.frequencies(act_mat=activity, adj_mat=self.adj_mat)
            phase_coherence = np.asarray([Kuramoto.phase_coherence(vec) for vec in activity.T])
            ####################################################################################################

            phase_vec_curr = activity.T[-1]
            last_freqs = frequencies.T[-1]

            mean_freqs = np.sum(frequencies * self.dt, axis=1) / self.t_step

            freqs_vec_curr = last_freqs
            # freqs_vec_curr = mean_freqs

            self.update_stats(_time=np.linspace(t_curr, t_curr + self.t_step, len(activity.T)),
                              _phases=np.sin(activity),
                              _frequencies=frequencies,
                              _phase_coherence=phase_coherence,
                              _time_=t_curr + self.t_step,
                              _last_freqs=last_freqs,
                              _mean_freqs=mean_freqs)
            self.N_lock.release()
            self.dt_lock.release()

            t_curr += self.t_step
            self.t_step_lock.release()

            # Redraw plots
            for i in range(3):
                _axes[i].clear()
            M = len(activity.T)
            _axes[0].set_ylabel(r'$\sin(\theta_i)$    ', rotation=0)
            _axes[0].plot(self.TIME[-10 * M:], self.PHASES.T[-10 * M:])
            _axes[1].set_ylabel('Coherence')
            _axes[1].plot(self.TIME[-100 * M:], self.PHASE_COHERENCE[-100 * M:], color='forestgreen')
            _axes[2].set_ylabel(r'$\overline{\omega}_i$    ', rotation=0)
            _axes[2].plot(self.TIME_[-100:], self.LAST_FREQS[-100:])

            # Критерий полной синхронизации
            if np.all(np.isclose(phase_coherence, 1.)):
                # print('\nПолная синхронизация!')
                showinfo(title='Сообщение', message='Полная синхронизация!')
                break

            # Критерий частичной синхронизации
            if len(self.MEAN_FREQS) > 20:
                cr1 = np.linalg.norm(np.asarray(self.MEAN_FREQS[-20:]) - mean_freqs)
                self.F_NORM.append(cr1)
                if len(self.F_NORM) > 20:
                    f_norm = np.asarray(self.F_NORM[-20:])
                    cr2 = np.linalg.norm(f_norm - np.mean(f_norm))
                    if cr1 < 0.1 or cr2 < 0.1:
                        # print('\nЧастичная синхронизация!')
                        showinfo(title='Сообщение', message='Частичная синхронизация!')
                        break

            _pause_event.wait()
            _canvas.draw()

    def show_graph(self, _canvas):
        pos = graphviz_layout(self.graph_nx, prog='dot', root=_canvas)
        nx.draw(self.graph_nx, pos, node_color='lightblue', edge_color='#909090', node_size=200, with_labels=True)
        _canvas.draw()
