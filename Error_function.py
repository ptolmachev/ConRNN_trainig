import autograd.numpy as np
from autograd import elementwise_grad as egrad

def fr_fun(x):
    return 1.0 / (1 + np.exp(-(x) / 50))# + 0.0001 * np.maximum(x, 0)

#error function for gradient checking
def Error_function(W_b, V_init, u_init, target, output_nrns, horizon):
    W = W_b[:W_b.shape[0] - 1, :].astype(np.float64)
    b = W_b[-1, :].astype(np.float64)
    V = V_init.astype(np.float64)
    u = u_init.astype(np.float64)
    dt = np.float64(0.2)
    alpha = np.float64(0.004)
    beta = np.float64(0.005)
    E = np.float64(0)
    for i in range(horizon):
        V_new = V + dt * ((fr_fun(V) @ W).flatten() + b - u)
        u_new = u + dt * alpha * (beta * V - u)
        V = V_new
        u = u_new
        s = fr_fun(V)
        E += np.sum((s[output_nrns] - target[i, output_nrns]) ** 2)
    return E
