# -*- coding: utf-8 -*-
import sys
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import kuramoto
from switch import Switch
from enum import Enum
import threading
from tkinter import *
from tkinter.messagebox import showinfo
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")
import os
os.environ["PATH"] += os.pathsep + os.path.join('C:\\', 'Program Files', 'Graphviz', 'bin')


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


if __name__ == '__main__':
    onn = ONN()

    root = Tk()

    root.title('Курамото v0.1')
    # root.geometry('{:.0f}x{:.0f}'.format(875, 575))
    root.resizable(width=False, height=False)

    N = IntVar(root, value=onn.N)
    K = DoubleVar(root, value=onn.K)
    dt = DoubleVar(root, value=onn.dt)
    t_step = DoubleVar(root, value=onn.t_step)
    T = DoubleVar(root, value=onn.T)
    p = DoubleVar(root, value=onn.p)
    seed = IntVar(root, value=onn.seed)

    main_menu = Menu(root)
    root.config(menu=main_menu)
    menu = [Menu(main_menu, tearoff=0) for _ in range(5)]

    # Топология
    topology = StringVar(root, value=onn.topology_mode.name)

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

    window_graph = None

    def show_graph():
        global window_graph
        window_graph = Toplevel(root)
        window_graph.title('Граф связей')
        fig, _ = plt.subplots(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(fig, master=window_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        onn.show_graph(canvas)

    menu[0].add_command(label='Предпросмотр...', command=lambda: show_graph())

    # Матрица смежности
    multiplier = StringVar(root, value=onn.multiplier_mode.name)

    menu[1].add_radiobutton(label=AdjMatMultiplierMode.NO.name,
                            variable=multiplier, value=AdjMatMultiplierMode.NO.name,
                            command=lambda: onn.__setattr__('multiplier_mode', AdjMatMultiplierMode.NO))
    menu[1].add_radiobutton(label=AdjMatMultiplierMode.RAND1.name,
                            variable=multiplier, value=AdjMatMultiplierMode.RAND1.name,
                            command=lambda: onn.__setattr__('multiplier_mode', AdjMatMultiplierMode.RAND1))
    menu[1].add_radiobutton(label=AdjMatMultiplierMode.RAND2.name,
                            variable=multiplier, value=AdjMatMultiplierMode.RAND2.name,
                            command=lambda: onn.__setattr__('multiplier_mode', AdjMatMultiplierMode.RAND2))
    # menu[1].add_separator()
    # menu[1].add_command(label='Тонкая настройка...')

    # Начальные фазы
    initial_phase = StringVar(root, value=onn.initial_phase_mode.name)

    menu[2].add_radiobutton(label=InitialPhaseMode.ZERO.name,
                            variable=initial_phase, value=InitialPhaseMode.ZERO.name,
                            command=lambda: onn.__setattr__('initial_phase_mode', InitialPhaseMode.ZERO))
    menu[2].add_radiobutton(label=InitialPhaseMode.RAND.name,
                            variable=initial_phase, value=InitialPhaseMode.RAND.name,
                            command=lambda: onn.__setattr__('initial_phase_mode', InitialPhaseMode.RAND))
    # menu[2].add_separator()
    # menu[2].add_command(label='Тонкая настройка...')

    # Собственные частоты
    initial_frequency = StringVar(root, value=onn.initial_frequency_mode.name)

    menu[3].add_radiobutton(label=InitialFrequencyMode.ZERO.name,
                            variable=initial_frequency, value=InitialFrequencyMode.ZERO.name,
                            command=lambda: onn.__setattr__('initial_frequency_mode', InitialFrequencyMode.ZERO))
    menu[3].add_radiobutton(label=InitialFrequencyMode.RAND1.name,
                            variable=initial_frequency, value=InitialFrequencyMode.RAND1.name,
                            command=lambda: onn.__setattr__('initial_frequency_mode', InitialFrequencyMode.RAND1))
    menu[3].add_radiobutton(label=InitialFrequencyMode.RAND2.name,
                            variable=initial_frequency, value=InitialFrequencyMode.RAND2.name,
                            command=lambda: onn.__setattr__('initial_frequency_mode', InitialFrequencyMode.RAND2))
    # menu[3].add_separator()
    # menu[3].add_command(label='Тонкая настройка...')

    # Другие параметры
    # draw_phases = BooleanVar(value=True)
    # draw_coherence = BooleanVar(value=True)
    # draw_frequencies = BooleanVar(value=True)
    save_plots = BooleanVar(value=False)

    # menu[4].add_checkbutton(label='Рисовать фазы', variable=draw_phases)
    # menu[4].add_checkbutton(label='Рисовать параметр порядка', variable=draw_coherence)
    # menu[4].add_checkbutton(label='Рисовать частоты', variable=draw_frequencies)
    # menu[4].add_separator()
    menu[4].add_checkbutton(label='Сохранять графики', variable=save_plots, state=DISABLED)

    main_menu.add_cascade(label='Топология', menu=menu[0])
    main_menu.add_cascade(label='Матрица смежности', menu=menu[1])
    main_menu.add_cascade(label='Начальные фазы', menu=menu[2])
    main_menu.add_cascade(label='Собственные частоты', menu=menu[3])
    main_menu.add_cascade(label='Настройки программы', menu=menu[4])

    spinbox_N = Spinbox(root, from_=4, to=sys.maxsize, increment=1, textvariable=N,
                        command=lambda: onn.__setattr__('N', N.get()))
    spinbox_K = Spinbox(root, from_=0.001, to=10, increment=0.001, textvariable=K,
                        command=lambda: onn.__setattr__('K', K.get()))
    spinbox_dt = Spinbox(root, from_=0.01, to=sys.maxsize, increment=0.01, textvariable=dt,
                         command=lambda: onn.__setattr__('dt', dt.get()))
    spinbox_t_step = Spinbox(root, from_=0.1, to=sys.maxsize, increment=0.1, textvariable=t_step,
                             command=lambda: onn.__setattr__('t_step', t_step.get()))
    spinbox_T = Spinbox(root, from_=0.1, to=sys.maxsize, increment=100, textvariable=T,
                        command=lambda: onn.__setattr__('T', T.get()))
    # spinbox_seed = Spinbox(root, from_=0, to=sys.maxsize, increment=1, textvariable=seed,
    #                        command=lambda: onn.__setattr__('seed', seed.get()))
    spinbox_p = Spinbox(root, from_=0.4, to=1, increment=0.01, textvariable=p,
                        command=lambda: onn.__setattr__('p', p.get()))

    spinbox_N.grid(row=0, column=2, padx=10, pady=10, sticky=W)
    spinbox_K.grid(row=1, column=2, padx=10, pady=10, sticky=W)
    spinbox_dt.grid(row=2, column=2, padx=10, pady=10, sticky=W)
    spinbox_t_step.grid(row=3, column=2, padx=10, pady=10, sticky=W)
    spinbox_T.grid(row=4, column=2, padx=10, pady=10, sticky=W)
    # spinbox_seed.grid(row=5, column=2, padx=10, pady=10, sticky=W)
    spinbox_p.grid(row=5, column=2, padx=10, pady=10, sticky=W)

    label_N = Label(root, text=' N ')
    label_K = Label(root, text=' K ')
    label_dt = Label(root, text=' dt ')
    label_t_step = Label(root, text=' t_step ')
    label_T = Label(root, text=' T ')
    # label_seed = Label(root, text=' seed ')
    label_p = Label(root, text=' p ')

    label_N.grid(row=0, column=0, padx=10, pady=10, sticky=E)
    label_K.grid(row=1, column=0, padx=10, pady=10, sticky=E)
    label_dt.grid(row=2, column=0, padx=10, pady=10, sticky=E)
    label_t_step.grid(row=3, column=0, padx=10, pady=10, sticky=E)
    label_T.grid(row=4, column=0, padx=10, pady=10, sticky=E)
    # label_seed.grid(row=5, column=0, padx=10, pady=10, sticky=E)
    label_p.grid(row=5, column=0, padx=10, pady=10, sticky=E)

    scale_N = Scale(root, orient=HORIZONTAL, from_=4, to=100, tickinterval=8, resolution=1, variable=N,
                    length=550, sliderlength=30,
                    command=lambda _: onn.__setattr__('N', N.get()))
    scale_K = Scale(root, orient=HORIZONTAL, from_=0.001, to=10, tickinterval=1, resolution=0.001, variable=K,
                    length=550, sliderlength=30,
                    command=lambda _: onn.__setattr__('K', K.get()))
    scale_dt = Scale(root, orient=HORIZONTAL, from_=0.01, to=0.5, tickinterval=0.05, resolution=0.01, variable=dt,
                     length=550, sliderlength=30,
                     command=lambda _: onn.__setattr__('dt', dt.get()))
    scale_t_step = Scale(root, orient=HORIZONTAL, from_=1, to=100, tickinterval=14, resolution=0.1, variable=t_step,
                         length=550, sliderlength=30,
                         command=lambda _: onn.__setattr__('t_step', t_step.get()))
    scale_T = Scale(root, orient=HORIZONTAL, from_=1000, to=150000, tickinterval=149000, resolution=1, variable=T,
                    length=550, sliderlength=30,
                    command=lambda _: onn.__setattr__('T', T.get()))
    # scale_seed = Scale(root, orient=HORIZONTAL, from_=0, to=1000, tickinterval=100, resolution=1, variable=seed,
    #                    length=550, sliderlength=30,
    #                    command=lambda _: onn.__setattr__('seed', seed.get()))
    scale_p = Scale(root, orient=HORIZONTAL, from_=0.4, to=1, tickinterval=0.1, resolution=0.001, variable=p,
                    length=550, sliderlength=30,
                    command=lambda _: onn.__setattr__('p', p.get()))

    scale_N.grid(row=0, column=1, padx=10, pady=10)
    scale_K.grid(row=1, column=1, padx=10, pady=10)
    scale_dt.grid(row=2, column=1, padx=10, pady=10)
    scale_t_step.grid(row=3, column=1, padx=10, pady=10)
    scale_T.grid(row=4, column=1, padx=10, pady=10)
    # scale_seed.grid(row=5, column=1, padx=10, pady=10)
    scale_p.grid(row=5, column=1, padx=10, pady=10)

    button_start = Button(root, text="Старт", width=20)
    button_start.grid(row=6, column=2, padx=10, pady=10)
    button_start.config(state=NORMAL)
    button_pause = Button(root, text="Пауза", width=20)
    button_pause.grid(row=6, column=1, padx=10, pady=10, sticky=E)
    button_pause.config(state=DISABLED)
    button_stop = Button(root, text="Стоп", width=20)
    button_stop.grid(row=6, column=1, padx=10, pady=10, sticky=W)
    button_stop.config(state=DISABLED)

    resume_event = threading.Event()
    resume_event.set()

    stop_event = threading.Event()

    window_simulation = None

    def start():
        stop_event.clear()

        global window_simulation
        window_simulation = Toplevel(root)
        window_simulation.title('Симуляция')
        fig, axes = plt.subplots(figsize=(10, 7), ncols=1, nrows=3)
        canvas = FigureCanvasTkAgg(fig, master=window_simulation)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        threading.Thread(target=onn.run,
                         args=(axes, canvas, resume_event, stop_event, save_plots.get(), ),
                         ).start()

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
        global window_simulation
        # noinspection PyUnresolvedReferences
        window_simulation.destroy()

    button_start.config(command=lambda: start())
    button_pause.config(command=lambda: pause())
    button_stop.config(command=lambda: stop())
    root.mainloop()

    plt.close()
