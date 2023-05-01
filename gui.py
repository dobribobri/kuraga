# -*- coding: utf-8 -*-
import sys
import itertools
from collections import defaultdict
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import kuramoto
from switch import Switch
import time
from enum import Enum
import threading
from tkinter import *
from matplotlib import pyplot as plt
import matplotlib as mpl


class Kuramoto(kuramoto.Kuramoto):
    def frequencies(self, act_mat, adj_mat):
        assert len(adj_mat) == act_mat.shape[0], 'adj_mat does not match act_mat'
        _, n_steps = act_mat.shape

        n_interactions = (adj_mat != 0).sum(axis=0)  # number of incoming interactions
        coupling = self.coupling / n_interactions  # normalize coupling by number of interactions

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for t in range(n_steps):
            dxdt[:, t] = self.derivative(act_mat[:, t], None, adj_mat, coupling=coupling)
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
                 _N: int = 7, _K: float = 0.4,
                 _p: float = 1.,
                 _seed: int = 42,

                 _topology_mode: TopologyMode = TopologyMode.FULL,
                 _adjmat_multiplier_mode: AdjMatMultiplierMode = AdjMatMultiplierMode.NO,
                 _initial_phase_mode: InitialPhaseMode = InitialPhaseMode.ZERO,
                 _initial_frequency_mode: InitialFrequencyMode = InitialFrequencyMode.ZERO,):

        self.dt, self.t_step, self.T = _dt, _t_step, _T
        self._N, self._K = _N, _K
        self._p = _p
        self._seed = _seed

        self._topology_mode = _topology_mode
        self.multiplier_mode = _adjmat_multiplier_mode

        self.initial_phase_mode = _initial_phase_mode
        self.initial_frequency_mode = _initial_frequency_mode

        self.graph_nx = self.new_graph_nx()
        self.adj_mat = self.new_adjmat()

        self.phase_vec_init = None
        self.nat_freqs_init = None
        self.initialize()

    def initialize(self):
        with Switch(self.initial_phase_mode) as case:
            if case(InitialPhaseMode.ZERO):
                phase_vec_init = np.zeros(self.N)
            if case(InitialPhaseMode.RAND):
                phase_vec_init = 2 * np.pi * np.random.random(size=self.N) - np.pi
            if case.default:
                phase_vec_init = np.zeros(self.N)
        self.phase_vec_init = phase_vec_init

        with Switch(self.nat_freqs_init) as case:
            if case(InitialFrequencyMode.ZERO):
                nat_freqs_init = np.zeros(self.N)
            if case(InitialFrequencyMode.RAND1):
                nat_freqs_init = np.random.random(size=self.N)
            if case(InitialFrequencyMode.RAND2):
                nat_freqs_init = 2 * np.random.random(size=self.N) - 1
            if case.default:
                nat_freqs_init = np.zeros(self.N)
        self.nat_freqs_init = nat_freqs_init

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
        assert _N > 0, ''
        self._N = _N
        self.graph_nx = self.new_graph_nx()
        if _N <= len(self.adj_mat):
            self.adj_mat = self.adj_mat[0:_N, 0:_N]
        else:
            adj_mat = self.new_adjmat()
            adj_mat[0:len(self.adj_mat), 0:len(self.adj_mat)] = self.adj_mat[:, :]
            self.adj_mat = adj_mat

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, _p: float):
        self.topology_mode = self._topology_mode

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, _seed: int):
        self._seed = _seed
        np.random.seed(_seed)

    def show_graph(self):
        pos = graphviz_layout(self.graph_nx, prog='dot')
        nx.draw(self.graph_nx, pos, node_color='lightblue', edge_color='#909090', node_size=200, with_labels=True)
        plt.show()

    def run(self, _pause_event, _stop_event):
        self.initialize()

        time_ = np.array([])
        angles_ = np.array([[]] * self.N)
        frequencies_ = np.array([[]] * self.N)
        phase_coherence_ = np.array([])

        t_curr = 0.
        natfreqs = self.nat_freqs_init
        angles_vec = self.phase_vec_init
        while (t_curr <= self.T) and not(_stop_event.is_set()):
            model = Kuramoto(coupling=1, dt=self.dt, T=self.t_step, n_nodes=self.N, natfreqs=natfreqs, )
            activity = model.run(adj_mat=self.adj_mat, angles_vec=angles_vec)
            frequencies = model.frequencies(act_mat=activity, adj_mat=self.adj_mat)
            phase_coherence = np.asarray([Kuramoto.phase_coherence(vec) for vec in activity.T])

            if np.all(np.isclose(phase_coherence, 1.)):
                break

            time_ = np.hstack((time_, np.arange(t_curr, t_curr + self.t_step, self.dt)))
            angles_ = np.hstack((angles_, np.sin(activity)))
            frequencies_ = np.hstack((frequencies_, frequencies))
            phase_coherence_ = np.hstack((phase_coherence_, phase_coherence))

            angles_vec = activity.T[-1]
            natfreqs = frequencies.T[-1]

            t_curr += self.t_step
            _pause_event.wait()


if __name__ == '__main__':
    onn = ONN()

    root = Tk()
    root.title('Курамото v0.1')
    root.geometry('{:.0f}x{:.0f}'.format(875, 650))
    root.resizable(width=False, height=False)

    N = IntVar(root, value=onn.N)
    K = DoubleVar(root, value=onn.K)
    dt = DoubleVar(root, value=onn.dt)
    t_step = DoubleVar(root, value=onn.t_step)
    T = DoubleVar(root, value=onn.T)
    topology = StringVar(root, value=onn.topology_mode.name)
    p = DoubleVar(root, value=onn.p)
    seed = IntVar(root, value=onn.seed)

    main_menu = Menu(root)
    root.config(menu=main_menu)
    menu = [Menu(main_menu, tearoff=0) for _ in range(5)]

    # Топология
    menu[0].add_radiobutton(label=TopologyMode.FULL.name,
                            variable=topology, value=TopologyMode.FULL.name,
                            command=lambda: onn.__setattr__('topology_mode', TopologyMode.FULL))
    menu[0].add_separator()
    menu[0].add_radiobutton(label=TopologyMode.STAR.name,
                            variable=topology, value=TopologyMode.STAR.name,
                            command=lambda: onn.__setattr__('topology_mode', TopologyMode.STAR))
    menu[0].add_radiobutton(label=TopologyMode.RING.name,
                            variable=topology, value=TopologyMode.RING.name,
                            command=lambda: onn.__setattr__('topology_mode', TopologyMode.RING))
    menu[0].add_radiobutton(label=TopologyMode.LINE.name,
                            variable=topology, value=TopologyMode.LINE.name,
                            command=lambda: onn.__setattr__('topology_mode', TopologyMode.LINE))
    menu[0].add_separator()
    menu[0].add_radiobutton(label=TopologyMode.RAND.name,
                            variable=topology, value=TopologyMode.RAND.name,
                            command=lambda: onn.__setattr__('topology_mode', TopologyMode.RAND))
    menu[0].add_separator()
    menu[0].add_command(label='Предпросмотр...', command=lambda: onn.show_graph())

    # Матрица смежности
    menu[1].add_radiobutton(label=AdjMatMultiplierMode.NO.name,
                            command=lambda: onn.__setattr__('multiplier_mode', AdjMatMultiplierMode.NO))
    menu[1].add_radiobutton(label=AdjMatMultiplierMode.RAND1.name,
                            command=lambda: onn.__setattr__('multiplier_mode', AdjMatMultiplierMode.RAND1))
    menu[1].add_radiobutton(label=AdjMatMultiplierMode.RAND2.name,
                            command=lambda: onn.__setattr__('multiplier_mode', AdjMatMultiplierMode.RAND2))
    menu[1].add_separator()
    menu[1].add_command(label='Тонкая настройка...')

    # Начальные фазы
    menu[2].add_radiobutton(label=InitialPhaseMode.ZERO.name,
                            command=lambda: onn.__setattr__('initial_phase_mode', InitialPhaseMode.ZERO))
    menu[2].add_radiobutton(label=InitialPhaseMode.RAND.name,
                            command=lambda: onn.__setattr__('initial_phase_mode', InitialPhaseMode.RAND))
    menu[2].add_separator()
    menu[2].add_command(label='Тонкая настройка...')

    # Собственные частоты
    menu[3].add_radiobutton(label=InitialFrequencyMode.ZERO.name,
                            command=lambda: onn.__setattr__('initial_frequency_mode', InitialFrequencyMode.ZERO))
    menu[3].add_radiobutton(label=InitialFrequencyMode.RAND1.name,
                            command=lambda: onn.__setattr__('initial_frequency_mode', InitialFrequencyMode.RAND1))
    menu[3].add_radiobutton(label=InitialFrequencyMode.RAND2.name,
                            command=lambda: onn.__setattr__('initial_frequency_mode', InitialFrequencyMode.RAND2))
    menu[3].add_separator()
    menu[3].add_command(label='Тонкая настройка...')

    # Другие параметры
    menu[4].add_checkbutton(label='Рисовать фазы')
    menu[4].add_checkbutton(label='Рисовать параметр порядка')
    menu[4].add_checkbutton(label='Рисовать частоты')
    menu[4].add_separator()
    menu[4].add_checkbutton(label='Сохранять графики')

    main_menu.add_cascade(label='Топология', menu=menu[0])
    main_menu.add_cascade(label='Матрица смежности', menu=menu[1])
    main_menu.add_cascade(label='Начальные фазы', menu=menu[2])
    main_menu.add_cascade(label='Собственные частоты', menu=menu[3])
    main_menu.add_cascade(label='Настройки программы', menu=menu[4])

    spinbox_N = Spinbox(root, from_=4, to=sys.maxsize, increment=1, textvariable=N,
                        command=lambda: onn.__setattr__('N', N.get()))
    spinbox_K = Spinbox(root, from_=0.01, to=100, increment=0.01, textvariable=K,
                        command=lambda: onn.__setattr__('K', K.get()))
    spinbox_dt = Spinbox(root, from_=0.01, to=sys.maxsize, increment=0.01, textvariable=dt,
                         command=lambda: onn.__setattr__('dt', dt.get()))
    spinbox_t_step = Spinbox(root, from_=0.1, to=sys.maxsize, increment=0.1, textvariable=t_step,
                             command=lambda: onn.__setattr__('t_step', t_step.get()))
    spinbox_T = Spinbox(root, from_=0.1, to=sys.maxsize, increment=100, textvariable=T,
                        command=lambda: onn.__setattr__('T', T.get()))
    spinbox_seed = Spinbox(root, from_=0, to=sys.maxsize, increment=1, textvariable=seed,
                           command=lambda: onn.__setattr__('seed', seed.get()))
    spinbox_p = Spinbox(root, from_=0.4, to=1, increment=0.01, textvariable=p,
                        command=lambda: onn.__setattr__('p', p.get()))

    spinbox_N.grid(row=0, column=2, padx=10, pady=10, sticky=W)
    spinbox_K.grid(row=1, column=2, padx=10, pady=10, sticky=W)
    spinbox_dt.grid(row=2, column=2, padx=10, pady=10, sticky=W)
    spinbox_t_step.grid(row=3, column=2, padx=10, pady=10, sticky=W)
    spinbox_T.grid(row=4, column=2, padx=10, pady=10, sticky=W)
    spinbox_seed.grid(row=5, column=2, padx=10, pady=10, sticky=W)
    spinbox_p.grid(row=6, column=2, padx=10, pady=10, sticky=W)

    label_N = Label(root, text=' N ')
    label_K = Label(root, text=' K ')
    label_dt = Label(root, text=' dt ')
    label_t_step = Label(root, text=' t_step ')
    label_T = Label(root, text=' T ')
    label_seed = Label(root, text=' seed ')
    label_p = Label(root, text=' p ')

    label_N.grid(row=0, column=0, padx=10, pady=10, sticky=E)
    label_K.grid(row=1, column=0, padx=10, pady=10, sticky=E)
    label_dt.grid(row=2, column=0, padx=10, pady=10, sticky=E)
    label_t_step.grid(row=3, column=0, padx=10, pady=10, sticky=E)
    label_T.grid(row=4, column=0, padx=10, pady=10, sticky=E)
    label_seed.grid(row=5, column=0, padx=10, pady=10, sticky=E)
    label_p.grid(row=6, column=0, padx=10, pady=10, sticky=E)

    scale_N = Scale(root, orient=HORIZONTAL, from_=4, to=100, tickinterval=8, resolution=1, variable=N,
                    length=550, sliderlength=30,
                    command=lambda _: onn.__setattr__('N', N.get()))
    scale_K = Scale(root, orient=HORIZONTAL, from_=0.01, to=100, tickinterval=10, resolution=0.01, variable=K,
                    length=550, sliderlength=30,
                    command=lambda _: onn.__setattr__('K', K.get()))
    scale_dt = Scale(root, orient=HORIZONTAL, from_=0.001, to=0.5, tickinterval=0.05, resolution=0.01, variable=dt,
                     length=550, sliderlength=30,
                     command=lambda _: onn.__setattr__('dt', dt.get()))
    scale_t_step = Scale(root, orient=HORIZONTAL, from_=1, to=100, tickinterval=14, resolution=0.1, variable=t_step,
                         length=550, sliderlength=30,
                         command=lambda _: onn.__setattr__('t_step', t_step.get()))
    scale_T = Scale(root, orient=HORIZONTAL, from_=1000, to=150000, tickinterval=149000, resolution=1, variable=T,
                    length=550, sliderlength=30,
                    command=lambda _: onn.__setattr__('T', T.get()))
    scale_seed = Scale(root, orient=HORIZONTAL, from_=0, to=1000, tickinterval=100, resolution=1, variable=seed,
                       length=550, sliderlength=30,
                       command=lambda _: onn.__setattr__('seed', seed.get()))
    scale_p = Scale(root, orient=HORIZONTAL, from_=0.4, to=1, tickinterval=0.1, resolution=0.001, variable=p,
                    length=550, sliderlength=30,
                    command=lambda _: onn.__setattr__('p', p.get()))

    scale_N.grid(row=0, column=1, padx=10, pady=10)
    scale_K.grid(row=1, column=1, padx=10, pady=10)
    scale_dt.grid(row=2, column=1, padx=10, pady=10)
    scale_t_step.grid(row=3, column=1, padx=10, pady=10)
    scale_T.grid(row=4, column=1, padx=10, pady=10)
    scale_seed.grid(row=5, column=1, padx=10, pady=10)
    scale_p.grid(row=6, column=1, padx=10, pady=10)

    button_start = Button(root, text="Старт", width=20)
    button_start.grid(row=7, column=2, padx=10, pady=10)
    button_start.config(state=NORMAL)
    button_pause = Button(root, text="Пауза", width=20)
    button_pause.grid(row=7, column=1, padx=10, pady=10, sticky=E)
    button_pause.config(state=DISABLED)
    button_stop = Button(root, text="Стоп", width=20)
    button_stop.grid(row=7, column=1, padx=10, pady=10, sticky=W)
    button_stop.config(state=DISABLED)

    resume_event = threading.Event()
    resume_event.set()

    stop_event = threading.Event()

    def start():
        stop_event.clear()
        threading.Thread(target=onn.run, args=(resume_event, stop_event, )).start()
        button_start.config(state=DISABLED)
        button_pause.config(state=NORMAL)
        button_stop.config(state=NORMAL)

    def pause():
        if resume_event.is_set():
            resume_event.clear()
            button_pause.config(text='Возобновить')
            button_start.config(state=DISABLED)
            button_pause.config(state=NORMAL)
            button_stop.config(state=DISABLED)
        else:
            resume_event.set()
            button_pause.config(text='Пауза')
            button_start.config(state=DISABLED)
            button_pause.config(state=NORMAL)
            button_stop.config(state=NORMAL)

    def stop():
        stop_event.set()
        button_start.config(state=NORMAL)
        button_pause.config(state=DISABLED)
        button_stop.config(state=DISABLED)

    button_start.config(command=lambda: start())
    button_pause.config(command=lambda: pause())
    button_stop.config(command=lambda: stop())
    root.mainloop()
