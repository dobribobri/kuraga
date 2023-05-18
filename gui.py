# -*- coding: utf-8 -*-
import os
import sys
from tkinter import *
from core import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib
matplotlib.use("TkAgg")


os.environ["PATH"] += os.pathsep + os.path.join('C:\\', 'Program Files', 'Graphviz', 'bin')


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
