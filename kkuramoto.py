
import kuramoto
import numpy as np


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
