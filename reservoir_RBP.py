import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt

class CRNN():
    def __init__(self, N, dt, params, V_init, u_init, weights, biases, histtory_len=50000, horizon=1000):
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
        self.horizon = horizon
        self.history_len = histtory_len
        self.rect_param = 0.0001
        self.V = V_init
        self.u = u_init
        self.p = np.zeros((self.N, self.N, self.N))
        self.q = np.zeros((self.N, self.N, self.N))
        self.r = np.zeros((self.N, self.N))
        self.l = np.zeros((self.N, self.N))
        self.t = 0
        self.fr_fun = lambda x: 1.0/(1 + np.exp(-(x - self.V_half) / self.slope)) + self.rect_param * np.maximum(x, self.V_half)
        self.fr_fun_der = lambda x: (1.0 / self.slope) * self.fr_fun(x) * (1 - self.fr_fun(x)) + self.rect_param * ((x - self.V_half) > 0)

        def inverse_fr_fun(y):
            y_ = y + 0.0001
            for i in range(5):
                x = - self.slope * np.log((1 - y_) / y_) + self.V_half
                y_ = y_ - self.rect_param * np.maximum(x, self.V_half)
            return x

        self.inverse_fr_fun = lambda y: inverse_fr_fun(y)

        # self.inverse_fr_fun = lambda x: - self.slope * np.log((1 - x) / x) + self.V_half
        # history
        self.V_history = deque(maxlen=self.history_len)
        self.target_history = deque(maxlen=self.history_len)
        self.b_history = deque(maxlen=self.history_len)
        self.t_range = deque(maxlen=self.history_len)
        # buffer for the error calculation
        self.V_buffer = deque(maxlen=horizon)
        self.u_buffer = deque(maxlen=horizon)
        self.p_buffer = deque(maxlen=horizon)
        self.q_buffer = deque(maxlen=horizon)
        self.r_buffer = deque(maxlen=horizon)
        self.l_buffer = deque(maxlen=horizon)

        # add initial values
        self.V_buffer.append(deepcopy(V_init))
        self.u_buffer.append(deepcopy(u_init))
        self.p_buffer.append(np.zeros((self.N, self.N, self.N)))
        self.q_buffer.append(np.zeros((self.N, self.N, self.N)))
        self.r_buffer.append(np.zeros((self.N, self.N)))
        self.l_buffer.append(np.zeros((self.N, self.N)))
        self.output_nrns = None
        self.target_signals = None

    def rhs_V(self):
        rhs_V = (self.fr_fun(self.V) @ self.W).flatten() + self.b - self.u #- self.alpha * self.V
        return rhs_V

    def rhs_u(self):
        rhs_u = self.alpha * (self.beta * self.V - self.u)
        return rhs_u

    def rhs_p(self):
        rhs_p = + np.einsum('kj,i', np.eye(self.N), self.fr_fun(self.V)) \
                + np.einsum('lk,l,lij', self.W, self.fr_fun_der(self.V), self.p) \
                - self.q #- self.alpha * self.p
        return rhs_p

    def rhs_q(self):
        rhs_q = self.alpha * (self.beta * self.p - self.q)
        return rhs_q

    def rhs_r(self):
        rhs_r = np.einsum('lk,l,li', self.W, self.fr_fun_der(self.V), self.r) + np.eye(self.N) - self.l
        return rhs_r

    def rhs_l(self):
        rhs_l = self.alpha * (self.beta * self.r - self.l)
        return rhs_l

    def get_next_state(self):
        V_next = self.V + self.dt * (self.rhs_V())
        u_next = self.u + self.dt * (self.rhs_u())
        return V_next, u_next

    def get_aux_variables(self):
        p_next = self.p + self.dt * (self.rhs_p())
        q_next = self.q + self.dt * (self.rhs_q())
        r_next = self.r + self.dt * (self.rhs_r())
        l_next = self.l + self.dt * (self.rhs_l())

        #RESERVOIR RELATED
        #set all variables to zero if it is not feedback or readout neurons:
        for j in range(self.N):
            if not j in self.output_nrns:
                p_next[:, :, j] = np.zeros((self.N, self.N))
                q_next[:, :, j] = np.zeros((self.N, self.N))
                r_next[:, j] = np.zeros(self.N)
                l_next[:, j] = np.zeros(self.N)

        return p_next, q_next, r_next, l_next

    def run(self, T_steps, record=False, save_every=None):
        for i in range(T_steps):
            next_V, next_u = self.get_next_state()
            self.V = deepcopy(next_V)
            self.u = deepcopy(next_u)
            self.t += self.dt

            if record == True:
                if save_every is None:
                    raise ValueError("One should also specify \'save_every\' parameter if \'record\' = True")
                if i % save_every == 0:
                    self.V_history.append(deepcopy(self.V))
                    self.t_range.append(deepcopy(self.t + self.dt))

    def run_and_learn(self, T_steps, update_every, lr_W=1e-4, lr_b=1e-6, teacher_forcing=False, record=False, save_every=None):
        for i in range(T_steps):
            V_next, u_next = self.get_next_state()
            p_next, q_next, r_next, l_next = self.get_aux_variables()

            self.V = deepcopy(V_next)
            self.u = deepcopy(u_next)
            self.p = deepcopy(p_next)
            self.q = deepcopy(q_next)
            self.r = deepcopy(r_next)
            self.l = deepcopy(l_next)
            self.t += self.dt

            #save in the buffer
            self.V_buffer.append(deepcopy(self.V))
            self.u_buffer.append(deepcopy(self.u))
            self.p_buffer.append(deepcopy(self.p))
            self.q_buffer.append(deepcopy(self.q))
            self.r_buffer.append(deepcopy(self.r))
            self.l_buffer.append(deepcopy(self.l))

            if (i % update_every == 0) and (i != 0):
                self.update_Wb(lr_W, lr_b, ind_start=i)
                # TEACHER FORCING
                # TODO MOdify TF to sent exact signal as feedback
                if teacher_forcing == True:
                    # reset the state V to the right values
                    self.V[self.output_nrns] = self.inverse_fr_fun(self.target_signals[i, :])
                    self.p = (np.zeros((self.N, self.N, self.N)))
                    self.q = (np.zeros((self.N, self.N, self.N)))
                    self.r = (np.zeros((self.N, self.N)))
                    self.l = (np.zeros((self.N, self.N)))

            if record == True:
                if save_every is None:
                    raise ValueError("One should also specify \'save_every\' parameter if \'record\' = True")
                if i % save_every == 0:
                    self.V_history.append(deepcopy(self.V))
                    self.b_history.append(deepcopy(self.b))
                    self.target_history.append(deepcopy(self.target_signals[i]))
                    self.t_range.append(deepcopy(self.t + self.dt))

        return None

    def update_Wb(self, lr_W, lr_b, ind_start):
        # calculate an error between the target signals and an actual output
        V_out = np.array(self.V_buffer)[:, self.output_nrns] #(t, o)
        p_out = np.array(self.p_buffer)[:, self.output_nrns, :, :] #(t, o, i, j)
        r_out = np.array(self.r_buffer)[:, self.output_nrns, :] #(t, o)

        #take horizon into account
        ind_end = ind_start + V_out.shape[0]
        e = (self.fr_fun(V_out) - self.target_signals[ind_start: ind_end, :])
        delta_W = - self.dt * lr_W * np.einsum("tk,tkij", e * self.fr_fun_der(V_out), p_out)
        #enforce zero self-coupling
        np.fill_diagonal(delta_W, 0)
        #RESERVOIR RELATED
        # enforce zero coupling between output neurons
        for i in (self.output_nrns):
            for j in (self.output_nrns):
                delta_W[i, j] = 0

        delta_b = - self.dt * lr_b * np.einsum("tk,tki", e * self.fr_fun_der(V_out), r_out)
        self.W = deepcopy(self.W + delta_W)
        self.b = deepcopy(self.b + delta_b)
        return None

    def set_target_signals(self, indices, target_signals):
        if len(indices) != (target_signals.shape[-1]):
            raise IOError('The length of indices and number of target signals should coincide!')
        self.output_nrns = indices

        #RESERVOIR RELATED
        # enforce zero coupling between output neurons
        for i in (self.output_nrns):
            for j in (self.output_nrns):
                self.W[i, j] = 0

        self.target_signals = target_signals
        return None

    def reset_history(self):
        self.t = 0
        self.V_history = deque(maxlen=self.history_len)
        self.target_history = deque(maxlen=self.history_len)
        self.b_history = deque(maxlen=self.history_len)
        self.t_range = deque(maxlen=self.history_len)
        # buffer for the error calculation
        self.V_buffer = deque(maxlen=self.horizon)
        self.u_buffer = deque(maxlen=self.horizon)
        self.p_buffer = deque(maxlen=self.horizon)
        self.q_buffer = deque(maxlen=self.horizon)
        self.r_buffer = deque(maxlen=self.horizon)
        self.l_buffer = deque(maxlen=self.horizon)

    def visualise_fr(self):
        V_array = np.array(self.V_history).T
        if not self.target_history is None: target_array = np.array(self.target_history).T
        k = 0
        # b_array = np.array(self.b_history).T
        # e = (1 / 4) * (self.target_signals[-V_array.shape[1]-1:-1, :].T - self.fr_fun(V_array[self.output_nrns])) ** 2
        t_array = np.array(self.t_range)
        fig, axes = plt.subplots(self.N, 1, figsize=(20, 10))
        if type(axes) != np.ndarray: axes = [axes]
        for i in range(len(axes)):
            if i == 0: axes[i].set_title('Firing Rates')
            axes[i].plot(t_array, self.fr_fun(V_array[i]), 'k', linewidth=2, alpha=0.9)

            if (not self.output_nrns is None) and (i in self.output_nrns):
                axes[i].plot(t_array, target_array[k], 'r', linewidth=2, alpha=0.3)
                k += 1

            axes[i].set_ylim([-0.1, 1.5])
            # axes[i].set_yticks([])
            # axes[i].set_yticklabels([])
            # axes[i].plot(t_array, V_array[i], 'r', linewidth=2, alpha=0.3)
            # axes[i].plot(t_array, b_array[i], 'b', linewidth=2, alpha=0.3)
            if i != len(axes) - 1:
                axes[i].set_xticks([])
                axes[i].set_xticklabels([])
            axes[i].set_xlabel('t, ms')
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.show()
        return None

if __name__ == '__main__':
    N = 20
    dt = 1
    params = dict()
    params['alpha'] = 0.001
    params['beta'] = 0.005
    params["V_half"] = 0.0
    params["slope"] = 50
    V_init = -50 + 100* np.random.rand(N)
    u_init = 0.02 * np.random.rand(N) - 0.01
    weights = 3 * np.random.rand(N, N) - 2
    biases = 0.1 + 0.05 * np.random.rand(N)
    rnn = CRNN(N, dt, params, V_init, u_init, weights, biases, horizon=300)

    # T_steps = 10000
    # rnn.run(T_steps, record=True, save_every=10)

    T_steps = 100000
    update_every = 100
    save_every = 1
    record = True

    #specify output_neurons and     a signal
    indices = [0]
    t_range = np.arange(200100) * dt
    fun = lambda x: (1.0 / (1 + np.exp(- (x + 30) / 30))) #+ 0.0001 * np.maximum(x, -30)

    # run with learning
    # target_signals = np.array([fun(-30 + 150 * np.sin(np.pi / 10 * t_range) + 200 * np.cos(((np.pi / 5) * t_range)) + 119 * np.sin(((np.pi / 15) * t_range) + np.pi/1.45) ),
    #                            fun(-29 + 50 * np.sin(np.pi / 7 * t_range) + 120 * np.cos(((np.pi / 14) * t_range)) + 132 * np.sin(((np.pi / 10.5) * t_range) + np.pi/1.25)),
    #                            fun(-35 + 140 * np.sin(np.pi / 11 * t_range) + 85 * np.cos(((np.pi / 5.5) * t_range)) + 119 * np.sin(((np.pi / 16.5) * t_range) + np.pi/2.43))]).T#,
    #                            # fun(-15 + 120 * np.sin(np.pi / 9.75 * t_range) + 123 * np.cos(((np.pi / 6.5) * t_range)) + 146 * np.sin(((np.pi / 13) * t_range) + np.pi/5.45))]).T

    # target_signals = np.array([0.25 * np.ones(len(t_range)), 0.75 * np.ones(len(t_range))]).T
    target_signals = np.array([
        fun(-30 + 150 * np.sin(np.pi / 900 * t_range))#,
        # fun(-30 + 150 * np.sin(np.pi / 1800 * t_range + np.pi))
    ]).T


    rnn.set_target_signals(indices, target_signals)
    rnn.run_and_learn(T_steps=T_steps, update_every=update_every, record=True,
                      lr_W=1e-6, lr_b=1e-6, teacher_forcing=True, save_every=save_every)

    rnn.visualise_fr()

    # # simple run
    # rnn.reset_history()
    # rnn.run(1000, record=record, save_every=save_every)
    # rnn.visualise_fr()


