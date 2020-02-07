import numpy as np
# import autograd.numpy as np
# from autograd import elementwise_grad as egrad
from collections import deque
from copy import deepcopy
from recurrent_net import CRNN
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from scipy import sparse

class LearningMechanism():
    '''
    Algorithm for learning in RNNs: rule for updating weights and biases during the run
    '''
    def __init__(self, RNN, params):
        '''
        :param RNN:
        :param params:
        :param fictive feedback: if True, the internal network sees the output neurons as if they are following the target trajectory

        '''
        self.RNN = RNN
        self.beta_1 = params['beta_1']
        self.beta_2 = params['beta_2']
        self.lr_W = params['lr_W']
        self.lr_b = params['lr_b']
        self.horizon = params['horizon']
        self.tf = params['teacher_forcing']
        self.fictive_feedback = params['fictive_feedback']
        self.len_error_history = params['len_error_history']
        self.update_lr = params['update_lr']
        self.error_buffer = deque(maxlen=self.len_error_history)
        self.V_buffer = deque(maxlen=self.horizon + 1)
        self.u_buffer = deque(maxlen=self.horizon + 1)
        self.V_buffer.append(deepcopy(self.RNN.V))
        self.u_buffer.append(deepcopy(self.RNN.u))
        self.target_history = deque(maxlen=self.RNN.history_len)
        # for Adam update
        self.m_W = np.zeros((self.RNN.N, self.RNN.N))
        self.m_b = np.zeros(self.RNN.N)
        self.v_W = np.zeros((self.RNN.N, self.RNN.N))
        self.v_b = np.zeros(self.RNN.N)

    def set_targets(self, out_nrns, targets):
        self.output_nrns = out_nrns
        self.targets = targets

    def rnn_step(self):
        self.RNN.run(1) #one time-step
        self.V_buffer.append(deepcopy(self.RNN.V))
        self.u_buffer.append(deepcopy(self.RNN.u))

    def calculate_error(self, desired):
        V_out = np.array(self.V_buffer)[1:, self.output_nrns]
        return np.linalg.norm(self.RNN.fr_fun(V_out) - desired, 2)

    def run_learning(self, T_steps):
        pass

    def calc_gradients(self, desired):
        pass


    def visualise(self):
        V_array = np.array(self.RNN.V_history).T
        target_array = np.array(self.target_history).T
        t_array = np.array(self.RNN.t_range)
        fig1, axes = plt.subplots(len(self.RNN.inds_record), 1, figsize=(20, 10))
        if type(axes) != np.ndarray: axes = [axes]
        k = 0
        r = 0
        for i, ind in enumerate(self.RNN.inds_record):
            if ind == 0: axes[i].set_title('Firing Rates')
            if ind in self.output_nrns:
                axes[i].plot(t_array, target_array[k], 'r', linewidth=2, alpha=0.5)
                k = k + 1
            axes[i].plot(t_array, self.RNN.fr_fun(V_array[r]), 'k', linewidth=2, alpha=0.9)
            axes[i].set_ylim([-0.1, 1.1])
            axes[i].set_yticks([])
            axes[i].set_yticklabels([])
            if i != len(axes) - 1:
                axes[i].set_xticks([])
                axes[i].set_xticklabels([])
            axes[i].set_xlabel('t, ms')
            r += 1
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.show()

        fig2 = plt.plot(figsize=(20, 10))
        plt.title("Error")
        plt.plot(np.array(self.error_buffer), 'r', linewidth=3)
        plt.show()

        return None

class BPTT(LearningMechanism):
    def __init__(self, RNN, params):
        super().__init__(RNN, params)

    def backprop(self, target):
        # Heavy stuff
        h = self.horizon
        N = self.RNN.N
        dt = self.RNN.dt
        W = self.RNN.W
        alpha = self.RNN.alpha
        beta = self.RNN.beta
        V_init = np.array(self.V_buffer)[0, :]
        # except initial conditions
        V_array = np.array(self.V_buffer)[1:, :].T # N x h
        target = target.T # N x h
        grad_W = np.zeros((N, N))
        grad_b = np.zeros((N))
        for p in np.arange(h)[::-1]:
            e = np.zeros(N)
            e[self.output_nrns] = 2 * (self.RNN.fr_fun(V_array[self.output_nrns, p]) - target[:, p])
            delta = self.RNN.fr_fun_der(V_array[:, p]) * e # delta on the last time step
            gamma = np.zeros_like(delta)
            for t in np.arange(p)[::-1]:
                grad_W += dt * deepcopy(self.RNN.fr_fun(V_array[:, t]).reshape(N, 1) @ delta.reshape(1, N))
                grad_b += dt * deepcopy(delta)
                delta_new = delta + dt * delta @ (W.T * self.RNN.fr_fun_der(V_array[:, t])) + dt * alpha * beta * gamma
                gamma_new = gamma - dt * (alpha * gamma + delta)
                delta = delta_new
                gamma = gamma_new
            # add the last piece: the gradient of weights from initial conditions to the first output
            grad_W += dt * deepcopy(self.RNN.fr_fun(V_init).reshape(N, 1) @ delta.reshape(1, N))
            grad_b += dt * deepcopy(delta)
        return grad_W, grad_b

    def calc_gradients(self, desired):
        # Use internal information from the buffer to calculate gradients
        # Calculate error term
        gradient_W, gradient_b = self.backprop(desired)
        grad_W = deepcopy(gradient_W)
        grad_b = deepcopy(gradient_b)
        return grad_W, grad_b

    def reset_buffers(self):
        self.V_buffer = deque(maxlen=self.horizon + 1)
        self.u_buffer = deque(maxlen=self.horizon + 1)
        self.V_buffer.append(deepcopy(self.RNN.V))
        self.u_buffer.append(deepcopy(self.RNN.u))
        return None

    def run_learning(self, T_steps):
        for i in tqdm(range(T_steps)):
            # once in a while:
            if (i != 0) and (i % self.horizon == 0):
                desired = self.targets[i - self.horizon:i, :]
                # Calculate weights' and biases' update
                grad_W, grad_b = self.calc_gradients(desired)
                # Enforce zero self coupling
                np.fill_diagonal(grad_W, 0)
                # Save new update directions
                self.v_W = self.mu * self.v_W - self.lr * grad_W
                self.v_b = self.mu * self.v_b - self.lr * grad_b
                # Apply changes
                self.RNN.W = self.RNN.W + self.v_W
                self.RNN.b = self.RNN.b + self.v_b

                if self.tf == True:
                    # Enforce current state to coincide with the target state
                    self.RNN.V[self.output_nrns] = deepcopy(self.RNN.inverse_fr_fun(self.targets[i, :]))

                # Reset buffers
                self.reset_buffers()

                # Decrese learning rate
                # self.lr *= 0.9995

            self.rnn_step()
            self.target_history.append(deepcopy(self.targets[i, :]))
        return None


class RealTimeRL(LearningMechanism):
    def __init__(self, RNN, params):
        super().__init__(RNN, params)
        # auxiliary buffers
        self.p_buffer = deque(maxlen=self.horizon + 1)
        self.q_buffer = deque(maxlen=self.horizon + 1)
        self.r_buffer = deque(maxlen=self.horizon + 1)
        self.l_buffer = deque(maxlen=self.horizon + 1)
        self.p = np.zeros((self.RNN.N, self.RNN.N, self.RNN.N), dtype = np.float64)
        self.q = np.zeros((self.RNN.N, self.RNN.N, self.RNN.N), dtype = np.float64)
        self.r = np.zeros((self.RNN.N, self.RNN.N), dtype = np.float64)
        self.l = np.zeros((self.RNN.N, self.RNN.N), dtype = np.float64)
        self.p_buffer.append(deepcopy(self.p))
        self.q_buffer.append(deepcopy(self.q))
        self.r_buffer.append(deepcopy(self.r))
        self.l_buffer.append(deepcopy(self.l))
        self.W_buffer = deque(maxlen=self.len_error_history)
        self.b_buffer = deque(maxlen=self.len_error_history)

    def rhs_p(self):
        # rhs_p = + np.einsum('ij,k,kj->jki', np.eye(self.RNN.N), self.RNN.fr_fun(self.RNN.V), np.ones((self.RNN.N, self.RNN.N)) - np.eye(self.RNN.N)) \
        rhs_p = + np.einsum('ij,k->jki', np.eye(self.RNN.N), self.RNN.fr_fun(self.RNN.V)) \
                + np.einsum('ij,i,ikl->jkl', self.RNN.W, self.RNN.fr_fun_der(self.RNN.V), self.p) \
                - self.q
        return rhs_p

    def rhs_q(self):
        rhs_q = self.RNN.alpha * (self.RNN.beta * self.p - self.q)
        return rhs_q

    def rhs_r(self):
        rhs_r = np.einsum('ij,i,ik->jk', self.RNN.W, self.RNN.fr_fun_der(self.RNN.V), self.r) + np.eye(self.RNN.N) - self.l
        return rhs_r

    def rhs_l(self):
        rhs_l = self.RNN.alpha * (self.RNN.beta * self.r - self.l)
        return rhs_l

    def get_aux_variables(self):
        p_next = self.p + self.RNN.dt * (self.rhs_p())
        q_next = self.q + self.RNN.dt * (self.rhs_q())
        r_next = self.r + self.RNN.dt * (self.rhs_r())
        l_next = self.l + self.RNN.dt * (self.rhs_l())
        return p_next, q_next, r_next, l_next

    def rnn_step(self, targets):
        # update aux variables befire V and u because they depend on the un-updeted variables
        p_next, q_next, r_next, l_next = self.get_aux_variables()
        self.p = deepcopy(p_next)
        self.q = deepcopy(q_next)
        self.r = deepcopy(r_next)
        self.l = deepcopy(l_next)
        # save in the buffer
        self.p_buffer.append(deepcopy(self.p))
        self.q_buffer.append(deepcopy(self.q))
        self.r_buffer.append(deepcopy(self.r))
        self.l_buffer.append(deepcopy(self.l))

        #weights and biases history

        if self.fictive_feedback == True:
            # run one step for output neurons:
            next_V_out = deepcopy(self.RNN.V[self.output_nrns] + self.RNN.dt * (self.RNN.rhs_V()[self.output_nrns]))
            next_u_out = deepcopy(self.RNN.u[self.output_nrns] + self.RNN.dt * (self.RNN.rhs_u()[self.output_nrns]))
            #set the output neurons as if they are following target trajectory
            self.RNN.V[self.output_nrns] = targets
            next_V_fictive = self.RNN.V + self.RNN.dt * (self.RNN.rhs_V())
            next_u_fictive = self.RNN.u + self.RNN.dt * (self.RNN.rhs_u())
            next_V_fictive[self.output_nrns] = next_V_out
            next_u_fictive[self.output_nrns] = next_u_out
            self.V_buffer.append(deepcopy(next_V_fictive))
            self.u_buffer.append(deepcopy(next_u_fictive))
            # update state of the RNN
            self.RNN.V = deepcopy(next_V_fictive)
            self.RNN.u = deepcopy(next_u_fictive)

            self.RNN.t += self.RNN.dt
            self.RNN.V_history.append(deepcopy(self.RNN.V))
            self.RNN.t_range.append(deepcopy(self.RNN.t + self.RNN.dt))
        else:
            self.RNN.run(1)  # one time-step
            self.V_buffer.append(deepcopy(self.RNN.V))
            self.u_buffer.append(deepcopy(self.RNN.u))
        return None

    def reset_buffers(self):
        # reset buffers
        self.V_buffer = deque(maxlen=self.horizon + 1)
        self.V_buffer.append(deepcopy(self.RNN.V))
        self.u_buffer = deque(maxlen=self.horizon + 1)
        self.u_buffer.append(deepcopy(self.RNN.u))

        # reset auxiliary buffers
        self.p_buffer = deque(maxlen=self.horizon + 1)
        self.q_buffer = deque(maxlen=self.horizon + 1)
        self.r_buffer = deque(maxlen=self.horizon + 1)
        self.l_buffer = deque(maxlen=self.horizon + 1)
        self.p = np.zeros((self.RNN.N, self.RNN.N, self.RNN.N))
        self.q = np.zeros((self.RNN.N, self.RNN.N, self.RNN.N))
        self.r = np.zeros((self.RNN.N, self.RNN.N))
        self.l = np.zeros((self.RNN.N, self.RNN.N))
        self.p_buffer.append(deepcopy(self.p))
        self.q_buffer.append(deepcopy(self.q))
        self.r_buffer.append(deepcopy(self.r))
        self.l_buffer.append(deepcopy(self.l))
        return None

    def calc_gradients(self, desired):
        # Take the arrays without the initial condition
        V_out = np.array(self.V_buffer)[1:, self.output_nrns]  # (time, outputs)
        p_out = np.array(self.p_buffer)[1:, self.output_nrns, :, :]  # (time, outputs, i, j)
        r_out = np.array(self.r_buffer)[1:, self.output_nrns, :]  # (time, outputs)
        e = 2 * (self.RNN.fr_fun(V_out) - desired)
        grad_W = np.einsum("ij,ijkl->kl", e * self.RNN.fr_fun_der(V_out), p_out)
        grad_b = np.einsum("ij,ijk->k", e * self.RNN.fr_fun_der(V_out), r_out)
        return grad_W, grad_b

    def run_learning(self, T_steps):
        for i in tqdm(range(T_steps)):
            #TODO: save initial position here
            if (i != 0) and (i % self.horizon == 0):
                desired = self.targets[i - self.horizon:i, :]
                # Calculate an error
                E = self.calculate_error(desired)
                self.error_buffer.append(E)
                # Calculate weights and biases gradients
                grad_W, grad_b = self.calc_gradients(desired)
                # Enforce zero self coupling
                np.fill_diagonal(grad_W, 0)

                # Save gradient's momenta
                self.m_W = self.beta_1 * self.m_W + (1 - self.beta_1) * grad_W
                self.m_b = self.beta_1 * self.m_b + (1 - self.beta_1) * grad_b
                self.v_W = self.beta_2 * self.v_W + (1 - self.beta_2) * (grad_W) ** 2
                self.v_b = self.beta_2 * self.v_b + (1 - self.beta_2) * (grad_b) ** 2

                # Apply changes
                eps = 0.0001
                self.RNN.W -= self.lr_W * (self.m_W / (1 - self.beta_1 ** (i / self.horizon))) / (np.sqrt(self.v_W) + eps)
                self.RNN.b -= self.lr_b * (self.m_b / (1 - self.beta_1 ** (i / self.horizon))) / (np.sqrt(self.v_b) + eps)

                self.W_buffer.append(deepcopy(self.RNN.W))
                self.b_buffer.append(deepcopy(self.RNN.b))

                if self.tf == True:
                    # Enforce current state to coincide with the target state
                    self.RNN.V[self.output_nrns] = deepcopy(self.RNN.inverse_fr_fun(self.targets[i, :]))
                # Reset buffers
                self.reset_buffers()
                # Change learning rate
                if self.update_lr == True:
                    self.lr_W *= 0.999
                    self.lr_b *= 0.999

            self.rnn_step(targets=self.RNN.inverse_fr_fun(self.targets[i, :]))
            self.target_history.append(deepcopy(self.targets[i, :]))
        return None


class ReservoirRLRL(LearningMechanism):
    def __init__(self, RNN, params, output_nrns):
        super().__init__(RNN, params)
        self.output_nrns = output_nrns
        self.v_W = np.zeros((self.RNN.N, len(self.output_nrns), 2))
        self.v_b = np.zeros(len(self.output_nrns))
        # check that there is no connections between output neurons
        for ind1 in self.output_nrns:
            for ind2 in self.output_nrns:
                self.RNN.W[ind1, ind2] = 0
        # no self-connections
        np.fill_diagonal(self.RNN.W, 0)

        # auxiliary buffers
        self.p_buffer = deque(maxlen=self.horizon + 1)
        self.q_buffer = deque(maxlen=self.horizon + 1)
        self.r_buffer = deque(maxlen=self.horizon + 1)
        self.l_buffer = deque(maxlen=self.horizon + 1)
        self.p = np.zeros((self.RNN.N, self.RNN.N, len(self.output_nrns), 2), dtype = np.float64)
        self.q = np.zeros((self.RNN.N, self.RNN.N, len(self.output_nrns), 2), dtype = np.float64)
        self.r = np.zeros((self.RNN.N, len(self.output_nrns)), dtype = np.float64)
        self.l = np.zeros((self.RNN.N, len(self.output_nrns)), dtype = np.float64)
        self.p_buffer.append(deepcopy(self.p))
        self.q_buffer.append(deepcopy(self.q))
        self.r_buffer.append(deepcopy(self.r))
        self.l_buffer.append(deepcopy(self.l))


    def rhs_p(self):
        # delta_ik s_j (1-delta_jk); k in self.output_neurons
        part1 = + np.einsum('ij,k,kj->ikj', np.eye(self.RNN.N)[:, :len(self.output_nrns)], self.RNN.fr_fun(self.RNN.V),
                            1 - np.eye(self.RNN.N)[:, :len(self.output_nrns)])
        rhs_p = + np.stack([part1, part1], axis=-1)\
                + np.einsum('ij,i,iklp->jklp', self.RNN.W, self.RNN.fr_fun_der(self.RNN.V), self.p) \
                - self.q
        return rhs_p

    def rhs_q(self):
        rhs_q = self.RNN.alpha * (self.RNN.beta * self.p - self.q)
        return rhs_q

    def rhs_r(self):
        mask = np.zeros((self.RNN.N,1))
        mask[self.output_nrns, :] = 1
        rhs_r = (self.RNN.W * self.RNN.fr_fun_der(self.RNN.V).reshape(self.RNN.N, 1)) @ self.r \
                 + np.ones_like(self.r) * mask \
                 - self.l
        return rhs_r

    def rhs_l(self):
        rhs_l = self.RNN.alpha * (self.RNN.beta * self.r - self.l)
        return rhs_l

    def get_aux_variables(self):
        p_next = self.p + self.RNN.dt * (self.rhs_p())
        q_next = self.q + self.RNN.dt * (self.rhs_q())
        r_next = self.r + self.RNN.dt * (self.rhs_r())
        l_next = self.l + self.RNN.dt * (self.rhs_l())
        return p_next, q_next, r_next, l_next

    def rnn_step(self, targets):
        # update aux variables befire V and u because they depend on the un-updeted variables
        p_next, q_next, r_next, l_next = self.get_aux_variables()
        self.p = deepcopy(p_next)
        self.q = deepcopy(q_next)
        self.r = deepcopy(r_next)
        self.l = deepcopy(l_next)
        # save in the buffer
        self.p_buffer.append(deepcopy(self.p))
        self.q_buffer.append(deepcopy(self.q))
        self.r_buffer.append(deepcopy(self.r))
        self.l_buffer.append(deepcopy(self.l))

        if self.fictive_feedback == True:
            # run one step for output neurons:
            next_V_out = deepcopy(self.RNN.V[self.output_nrns] + self.RNN.dt * (self.RNN.rhs_V()[self.output_nrns]))
            next_u_out = deepcopy(self.RNN.u[self.output_nrns] + self.RNN.dt * (self.RNN.rhs_u()[self.output_nrns]))
            #set the output neurons as if they are following target trajectory
            self.RNN.V[self.output_nrns] = targets
            next_V_fictive = self.RNN.V + self.RNN.dt * (self.RNN.rhs_V())
            next_u_fictive = self.RNN.u + self.RNN.dt * (self.RNN.rhs_u())
            next_V_fictive[self.output_nrns] = next_V_out
            next_u_fictive[self.output_nrns] = next_u_out
            self.V_buffer.append(deepcopy(next_V_fictive))
            self.u_buffer.append(deepcopy(next_u_fictive))
            # update state of the RNN
            self.RNN.V = deepcopy(next_V_fictive)
            self.RNN.u = deepcopy(next_u_fictive)

            self.RNN.t += self.RNN.dt
            self.RNN.V_history.append(deepcopy(self.RNN.V))
            self.RNN.t_range.append(deepcopy(self.RNN.t + self.RNN.dt))
        else:
            self.RNN.run(1)  # one time-step
            self.V_buffer.append(deepcopy(self.RNN.V))
            self.u_buffer.append(deepcopy(self.RNN.u))
        return None

    def reset_buffers(self):
        # reset buffers
        self.V_buffer = deque(maxlen=self.horizon + 1)
        self.V_buffer.append(deepcopy(self.RNN.V))
        self.u_buffer = deque(maxlen=self.horizon + 1)
        self.u_buffer.append(deepcopy(self.RNN.u))

        # reset auxiliary buffers
        self.p_buffer = deque(maxlen=self.horizon + 1)
        self.q_buffer = deque(maxlen=self.horizon + 1)
        self.r_buffer = deque(maxlen=self.horizon + 1)
        self.l_buffer = deque(maxlen=self.horizon + 1)
        self.p = np.zeros((self.RNN.N, self.RNN.N, len(self.output_nrns), 2), dtype = np.float64)
        self.q = np.zeros((self.RNN.N, self.RNN.N, len(self.output_nrns), 2), dtype = np.float64)
        self.r = np.zeros((self.RNN.N, len(self.output_nrns)), dtype = np.float64)
        self.l = np.zeros((self.RNN.N, len(self.output_nrns)), dtype = np.float64)
        self.p_buffer.append(deepcopy(self.p))
        self.q_buffer.append(deepcopy(self.q))
        self.r_buffer.append(deepcopy(self.r))
        self.l_buffer.append(deepcopy(self.l))
        return None

    def calc_gradients(self, desired):
        # Take the arrays without the initial condition
        V_out = np.array(self.V_buffer)[1:, self.output_nrns]  # (time, outputs)
        p_out = np.array(self.p_buffer)[1:, self.output_nrns, :, :, :]  # (time, outputs, i, j)
        r_out = np.array(self.r_buffer)[1:, self.output_nrns, :]  # (time, outputs)
        e = 2 * (self.RNN.fr_fun(V_out) - desired)
        # ti, tij
        d_W = np.einsum("ij,ijklp->klp", e * self.RNN.fr_fun_der(V_out), p_out)
        grad_b = np.einsum("ij,ijk->k", e * self.RNN.fr_fun_der(V_out), r_out)
        return d_W, grad_b

    def run_learning(self, T_steps):
        for i in tqdm(range(T_steps)):
            if (i != 0) and (i % self.horizon == 0):
                desired = self.targets[i - self.horizon:i, :]
                # Calculate an error
                E = self.calculate_error(desired)
                self.error_buffer.append(E)
                # Calculate weights and biases gradients
                d_W, grad_b = self.calc_gradients(desired)
                # Save new update directions
                self.v_W = self.mu * self.v_W - self.lr * d_W
                self.v_b = self.mu * self.v_b - self.lr * grad_b
                # Apply changes
                self.RNN.W[:, self.output_nrns] = self.v_W[:, :, 0]
                self.RNN.W[self.output_nrns, :] = self.v_W[:, :, 1].T
                self.RNN.b = self.RNN.b + self.v_b
                if self.tf == True:
                    # Enforce current state to coincide with the target state
                    self.RNN.V[self.output_nrns] = deepcopy(self.RNN.inverse_fr_fun(self.targets[i, :]))
                # Reset buffers
                self.reset_buffers()
                # Change learning rate
                if self.update_lr == True:
                    self.lr = 0.002 * np.sqrt(E)
            self.rnn_step(targets=self.RNN.inverse_fr_fun(self.targets[i, :]))
            self.target_history.append(deepcopy(self.targets[i, :]))
        return None


if __name__ == '__main__':
    N = 2
    dt = 1
    T_steps = 200000
    save_every = 1
    record = True
    inds_record = np.arange(np.minimum(N, 10))
    params = dict()
    params['alpha'] = 0.0015
    params['beta'] = 0.005
    params["V_half"] = 0.0
    params["slope"] = 50
    V_init = -50 + 100 * np.random.rand(N)
    u_init = 0.02 * np.random.rand(N) - 0.01
    # set sparse connectivity
    sparsity = 0.1
    M = np.sign(np.maximum(0, np.random.rand(N, N) - (1 - sparsity)))
    np.fill_diagonal(M, 0)
    # weights = ((0.05 - 0.3 * np.random.randn(N, N)) * M)
    weights = -1. * np.random.rand(N, N) + 0.4
    biases = 0.1 + 0.1 * np.random.rand(N)
    rnn = CRNN(N, dt, params, V_init, u_init, weights, biases, record=record,inds_record=inds_record, save_every=save_every)
    params_lm = dict()
    params_lm['len_error_history'] = 500000
    params_lm['lr_b'] = 1e-2
    params_lm['lr_W'] = 1e-2
    params_lm['beta_1'] = 0.9
    params_lm['beta_2'] = 0.999
    params_lm['horizon'] = 100
    params_lm['teacher_forcing'] = False
    params_lm['fictive_feedback'] = False
    params_lm['update_lr'] = True
    # lm = BPTT(RNN=rnn, params=params_lm)
    lm = RealTimeRL(RNN=rnn, params=params_lm)
    out_nrns = [0]
    # lm = ReservoirRLRL(RNN=rnn, params=params_lm, output_nrns=out_nrns)
    t_range = np.arange(T_steps + 2 * params_lm['horizon'])
    targets = np.array([
                        # 0.25 * np.ones(len(t_range)),
                        # 0.75 * np.ones(len(t_range))
                        # rnn.fr_fun(-30 + 150 * np.sin(np.pi / 1200 * t_range) + 100 * np.cos(np.pi / 1800 * t_range))#,
                        rnn.fr_fun(350 * np.sin(np.pi / 4000 * t_range + np.pi))
                        ]).T
    lm.set_targets(out_nrns, targets)
    lm.run_learning(T_steps)
    lm.visualise()

    fig, axs = plt.subplots(2, 1, figsize=(20,10))
    axs[0].plot(np.array(lm.W_buffer).reshape(len(lm.W_buffer), -1))
    axs[1].plot(np.array(lm.b_buffer))
    plt.show()

    # simple run
    rnn.reset_history()
    rnn.run(T_steps)
    rnn.visualise()


