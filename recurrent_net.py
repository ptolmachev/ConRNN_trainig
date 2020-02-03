import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt

class CRNN():
    def __init__(self, N, dt, params, V_init, u_init, weights, biases, record, save_every,  history_len=50000):
        self.N = N
        self.dt = dt
        self.params = params
        # enforce zero self-coupling
        np.fill_diagonal(weights, 0)
        self.W = weights
        self.b = biases
        self.V_half = params['V_half']
        self.slope = params['slope']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.history_len = history_len
        self.record = record
        self.save_every = save_every
        self.V = V_init
        self.u = u_init
        self.p = np.zeros((self.N, self.N, self.N))
        self.q = np.zeros((self.N, self.N, self.N))
        self.r = np.zeros((self.N, self.N))
        self.l = np.zeros((self.N, self.N))
        self.t = 0

        self.rect_param = 0.0001
        self.fr_fun = lambda x: 1.0/(1 + np.exp(-(x - self.V_half) / self.slope))# + self.rect_param * np.maximum(x, self.V_half)
        self.fr_fun_der = lambda x: (1.0 / self.slope) * self.fr_fun(x) * (1 - self.fr_fun(x))# + self.rect_param * ((x - self.V_half) > 0)

        # def inverse_fr_fun(y):
        #     y_ = y + 0.0001
        #     for i in range(5):
        #         x = - self.slope * np.log((1 - y_) / y_) + self.V_half
        #         y_ = y_ - self.rect_param * np.maximum(x, self.V_half)
        #     return x
        # self.inverse_fr_fun = lambda y: inverse_fr_fun(y)

        self.inverse_fr_fun = lambda y: - self.slope * np.log((1 - y) / y) + self.V_half
        if self.record == True:
            self.V_history = deque(maxlen=self.history_len)
            self.target_history = deque(maxlen=self.history_len)
        self.t_range = deque(maxlen=self.history_len)


    def rhs_V(self):
        rhs_V = (self.fr_fun(self.V) @ self.W).flatten() + self.b - self.u
        return rhs_V

    def rhs_u(self):
        rhs_u = self.alpha * (self.beta * self.V - self.u)
        return rhs_u

    def get_next_state(self):
        V_next = self.V + self.dt * (self.rhs_V())
        u_next = self.u + self.dt * (self.rhs_u())
        return V_next, u_next


    def run(self, T_steps):
        for i in range(T_steps):
            next_V, next_u = self.get_next_state()
            self.V = deepcopy(next_V)
            self.u = deepcopy(next_u)
            self.t += self.dt

            if self.record == True:
                if self.save_every is None:
                    raise ValueError("One should also specify \'save_every\' parameter if \'record\' = True")
                if i % self.save_every == 0:
                    self.V_history.append(deepcopy(self.V))
                    self.t_range.append(deepcopy(self.t + self.dt))
        return None

    def reset_history(self):
        self.t = 0
        self.V_history = deque(maxlen=self.history_len)
        self.target_history = deque(maxlen=self.history_len)
        self.t_range = deque(maxlen=self.history_len)
        return None

    def visualise_fr(self):
        V_array = np.array(self.V_history).T
        t_array = np.array(self.t_range)
        fig, axes = plt.subplots(self.N, 1, figsize=(20, 10))
        if type(axes) != np.ndarray: axes = [axes]
        for i in range(len(axes)):
            if i == 0: axes[i].set_title('Firing Rates')
            axes[i].plot(t_array, self.fr_fun(V_array[i]), 'k', linewidth=2, alpha=0.9)
            axes[i].set_ylim([-0.1, 1.2])
            # axes[i].set_yticks([])
            # axes[i].set_yticklabels([])
            if i != len(axes) - 1:
                axes[i].set_xticks([])
                axes[i].set_xticklabels([])
            axes[i].set_xlabel('t, ms')
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.show()
        return None

if __name__ == '__main__':
    N = 10
    dt = 1
    T_steps = 100000
    save_every = 1
    record = True
    params = dict()
    params['alpha'] = 0.0015
    params['beta'] = 0.005
    params["V_half"] = 0.0
    params["slope"] = 50
    V_init = -50 + 100* np.random.rand(N)
    u_init = 0.02 * np.random.rand(N) - 0.01
    weights = 3 * np.random.rand(N, N) - 2
    biases = 0.1 + 0.05 * np.random.rand(N)
    rnn = CRNN(N, dt, params, V_init, u_init, weights, biases, record=record, save_every=save_every)

    # simple run
    rnn.reset_history()
    rnn.run(T_steps)
    rnn.visualise_fr()


