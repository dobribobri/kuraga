
import numpy as np


class Stats:
    def __init__(self, _N):
        self._N = _N
        self.TIME, self.PHASES, self.FREQUENCIES, self.PHASE_COHERENCE = None, None, None, None
        self.TIME_, self.MEAN_FREQS, self.LAST_FREQS, self.F_NORM = None, None, None, None
        self.clear()

    def clear(self):
        self.TIME = np.array([])
        self.PHASES = np.array([[]] * self._N)
        self.FREQUENCIES = np.array([[]] * self._N)
        self.PHASE_COHERENCE = np.array([])
        self.TIME_ = []
        self.MEAN_FREQS = []
        self.LAST_FREQS = []
        self.F_NORM = []

    def update(self, _time, _phases, _frequencies, _phase_coherence,
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
