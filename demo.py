"""
AI-Empowered Medicine and Healthcare Homework 1 
Author: YH G; QM W.

Sparse Coding for noisy data

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run demo.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
    (4) Save results for the four experiments by pickle with name "results_taskID.pkl".
Student name(s):
Student ID(s):
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
# Don't add any other packages


def matching_pursuit(x: np.array, psi: np.array, s: int) -> np.array:
    """
    Matching pursuit algorithm
    :param x: a signal with size (N, 1)
    :param psi: a dictionary with size (N, M)
    :param s: the maximum number of nonzero elements
    :return:
        a: a sparse coefficient vector with size (M, 1)

    TODO: Implement MP algorithm below
    """
    r = x.copy()
    N, M = psi.shape
    a = np.zeros((M, 1))
    for n in range(s):
        epslion = np.transpose(psi) @ r
        i = np.argmax(np.abs(epslion))
        r = r / np.linalg.norm(r)
        psi[:, i] = psi[:, i]/ np.linalg.norm(psi[:, i])
        a[i] = np.transpose(r) @ psi[:,i]
        r = r - np.transpose([a[i] * psi[:, i]])
    return a


def orthogonal_matching_pursuit(x: np.array, psi: np.array, s: int) -> np.array:
    """
    Orthogonal matching pursuit algorithm
    :param x: a signal with size (N, 1)
    :param psi: a dictionary with size (N, M)
    :param s: the maximum number of nonzero elements
    :return:
        a: a sparse coefficient vector with size (M, 1)

    TODO: Implement OMP algorithm below
    """
    N, M = psi.shape
    S = np.zeros((N, M))
    r = x.copy()
    for n in range(s):
        epslion = np.transpose(psi) @ r
        i = np.argmax(np.abs(epslion))
        S[:, i] = 1
        a = l2_regularized_method(x, psi * S)
        r = x - (psi * S) @ a
    return a

def basis_pursuit(x: np.array, psi: np.array, gamma: float, max_iter: int) -> np.array:
    # """
    # Basis pursuit algorithm
    # :param x: a signal with size (N, 1)
    # :param psi: a dictionary with size (N, M)
    # :param gamma: the weight of the l1-norm regularization
    # :param max_iter: the maximum number of iterations
    # :return:
    #     a: a sparse coefficient vector with size (M, 1)

    # TODO: Implement BP (soft-thresholding-based) algorithm below
    # TODO: Note that besides max_iter, you can add other hyperparameter to define more flexible stop criterion
    
    # # https://math.stackexchange.com/questions/471339/derivation-of-soft-thresholding-operator-proximal-operator-of-l-1-norm
    # """
    
    # pre-compute: psipsi = psi^T @ psi
    psipsi = psi.T @ psi # (200, 200)
    lr = 0.00005
    s = np.zeros((psi.shape[1], 1))
    def apply_grad_descent(a, x, lr=lr, s=s):
        grad = - (psi.T @ x - psipsi @ a)  # issue: grad might be very large
        # print(grad[:10])

        # s = 0.99 * s + 0.01 * grad * grad
        s = 0.999 * s + 0.001 * grad * grad
        # s = grad * grad
        grad = grad / np.sqrt(np.sum(s))
        ret = a - lr * grad
        return ret
    
    def prox(x, gamma=gamma, lr=lr):
        # prox(x) = argmin_z(\| x - z \|^2 + 2 * lr * \gamma * |z|)
        # <=> prox(x)_i = sign(x_i)(|x_i|-lr*\gamma)
        z = np.sign(x) * np.maximum((np.abs(x) - lr*gamma), 0)
        # print("+:{};zero:{};-:{}".format((z>0).sum(), (z==0).sum(), (z<0).sum()))
        return z

    gamma_len = psi.shape[1]
    a = np.ones((gamma_len, 1)) / gamma_len  # Initialize a

    for epoch in range(max_iter):
        # import ipdb; ipdb.set_trace()
        # print('------{}-------'.format(epoch))
        _a = apply_grad_descent(a, x, lr=lr,s=s)
        # print(_a[:10])
        if abs(_a[0]) > 1e10:
            assert False
        # import ipdb; ipdb.set_trace()
        a = prox(_a, gamma, lr)
        # print(a[:10])
        # import ipdb; ipdb.set_trace()
    return a


def l0_regularized_method(x: np.array, psi: np.array, gamma: float, max_iter: int) -> np.array:
    # """
    # Basis pursuit algorithm
    # :param x: a signal with size (N, 1)
    # :param psi: a dictionary with size (N, M)
    # :param gamma: the weight of the l1-norm regularization
    # :param max_iter: the maximum number of iterations
    # :return:
    #     a: a sparse coefficient vector with size (M, 1)

    # TODO: Implement BP (soft-thresholding-based) algorithm below
    # TODO: Note that besides max_iter, you can add other hyperparameter to define more flexible stop criterion
    
    # # https://math.stackexchange.com/questions/471339/derivation-of-soft-thresholding-operator-proximal-operator-of-l-1-norm
    # """
    
    # pre-compute: psipsi = psi^T @ psi
    import math
    psipsi = psi.T @ psi # (200, 200)
    lr = 0.00001
    s_sq = np.zeros((psi.shape[1], 1))
    # s = 0
    def apply_grad_descent(a, x, momentum, lr=lr, s_sq=s_sq):
        grad = - (psi.T @ x - psipsi @ a)  # issue: grad might be very large
        s_sq = 0.999 * s_sq + 0.001 * grad * grad
        grad = grad / np.sqrt(np.sum(s_sq)) 
        # grad = grad/ np.sqrt(s_sq)
        # import ipdb; ipdb.set_trace()
        # grad = 0.75 * grad + 0.15 * noise + 0.1 * momentum
        mask = np.random.randint(0,2,grad.shape)
        # grad  = grad * mask + momentum * (1-mask)
        grad  = grad + momentum * (1-mask)
        
        # noise = np.random.random(size=grad.shape) 
        # grad = 0.9*grad + 0.1*noise
        noise = np.random.randn(*grad.shape)*lr/10  # gaussian
        grad  = grad + noise

        # grad = grad / np.sqrt(s_sq)
        ret = a - lr * grad
        return ret
    
    def prox(x, gamma=gamma, lr=lr):
        z = np.where(np.abs(x)>np.sqrt(2*lr*gamma), x, 0)
        return z

    gamma_len = psi.shape[1]
    a = np.ones((gamma_len, 1)) / gamma_len  # Initialize a

    _m = 0
    for epoch in range(max_iter):
        # print('------{}-------'.format(epoch))
        # import ipdb; ipdb.set_trace()
        _lr = lr + lr*(1-math.log((epoch+1)/max_iter+1))
        _a = apply_grad_descent(a, x, _m, lr=_lr,s_sq=s_sq)
        _m = _a
        # print(_a[:10])
        if abs(_a[0]) > 1e10:
            assert False
        # import ipdb; ipdb.set_trace()
        a = prox(_a, gamma, lr)
        # print(a[:10])
        # import ipdb; ipdb.set_trace()
    return a



def _l0_regularized_method(x: np.array, psi: np.array, gamma: float, max_iter: int) -> np.array:
    """
    Learning sparse codes with l0-norm regularization by hard-thresholding
    :param x: a signal with size (N, 1)
    :param psi: a dictionary with size (N, M)
    :param gamma: the weight of the l1-norm regularization
    :param max_iter: the maximum number of iterations
    :return:
        a: a sparse coefficient vector with size (M, 1)

    TODO: Implement hard-thresholding-based algorithm below
    TODO: Note that besides max_iter, you can add other hyperparameter to define more flexible stop criterion
    """

     # pre-compute: psipsi = psi^T @ psi
    psipsi = psi.T @ psi # (200, 200)
    lr = 0.00005
    s = np.zeros((200, 1))
    def apply_grad_descent(a, x, lr=lr, s=s):
        grad = - (psi.T @ x - psipsi @ a)  # issue: grad might be very large
        # print(grad[:10])
        # s = 0.99 * s + 0.01 * grad * grad
        s = 0.999 * s + 0.001 * grad * grad
        # s = grad * grad
        grad = grad / np.sqrt(np.sum(s))
        ret = a - lr * grad
        return ret
    
    def prox(x, gamma=gamma, lr=lr):
        # prox(x) = argmin_z(\| x - z \|^2 + 2 * lr * \gamma * |z|)
        # <=> prox(x)_i = sign(x_i)(|x_i|-lr*\gamma)
        z = np.sign(x) * np.maximum((np.abs(x) - lr*gamma), 0)
        # z = np.where(x>np.sqrt(2*lr*gamma), 0, x)
        # print("+:{};zero:{};-:{}".format((z>0).sum(), (z==0).sum(), (z<0).sum()))
        return z

    gamma_len = psi.shape[1]
    a = np.ones((gamma_len, 1)) / gamma_len  # Initialize a

    for epoch in range(max_iter):
        # import ipdb; ipdb.set_trace()
        # print('------{}-------'.format(epoch))
        _a = apply_grad_descent(a, x, lr=lr,s=s)
        # print(_a[:10])
        if abs(_a[0]) > 1e10:
            assert False
        # import ipdb; ipdb.set_trace()
        a = prox(_a, gamma, lr)
        # print(a[:10])
        # import ipdb; ipdb.set_trace()


def admm_method(x: np.array, psi: np.array, gamma: float, max_iter: int) -> np.array:
    """
    ADMM-based algorithm
    :param x: a signal with size (N, 1)
    :param psi: a dictionary with size (N, M)
    :param gamma: the weight of the l1-norm regularization
    :param max_iter: the maximum number of iterations
    :return:
        a: a sparse coefficient vector with size (M, 1)

    TODO: Implement the ADMM-based algorithm below
    TODO: Note that besides max_iter, you can add other hyperparameter to define more flexible stop criterion
    # YH note: http://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/admm.pdf
    # http://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/prox-grad.pdf
    """

    
    def prox(x, gamma, lr):
        z = np.sign(x) * np.maximum((np.abs(x) - lr*gamma), 0)
        return z

    gamma_len = psi.shape[1]
    a = np.ones((gamma_len, 1)) / gamma_len  # Initialize a
    y = a.copy()
    z = np.zeros((gamma_len, 1))
    rau = 0.1
    psipsi = psi.T @ psi # (200, 200)
    # lr = 0.00005
    lr = 0.01
    for epoch in range(max_iter):
        # 1. Apply Least-Square Estimation on alpha
        _a_1 = (psipsi + rau*np.eye(psi.shape[1]))
        _a_1 = np.linalg.inv(_a_1)
        _a_2 = psi.T @ x + rau * (y - z)
        a = _a_1 @ _a_2
        # 2. Apply soft thresolding to y
        y = a + z
        y = prox(y, gamma, lr)
        # 3. Update z
        z = z + a - y
    return a



def l2_regularized_method(x: np.array, psi: np.array, gamma: float = 1e-3) -> np.array:
    """
    The baseline method solving min_a ||x - psi * a||_2^2 + gamma * ||a||_2^2
    :param x: a signal with size (N, 1)
    :param psi: a dictionary with size (N, M)
    :param gamma: the weight of the l1-norm regularization
    :return:
        a: a sparse coefficient vector with size (M, 1)
    """
    a_mat = np.transpose(psi) @ psi + gamma * np.eye(psi.shape[1])
    b_vec = np.transpose(psi) @ x
    return np.linalg.solve(a_mat, b_vec)


def generate_dictionary(dim: int = 100) -> np.array:
    psi1 = np.zeros((dim, dim))
    psi2 = np.zeros((dim, dim))
    ns = np.arange(dim)
    for n in range(dim):
        psi1[:, n] = np.cos(np.pi / dim * n * (0.5 + ns))
        psi2[:, n] = 0.5 * (np.sign(np.sin(np.pi * (n / dim) * ns)) + 1)
    return np.concatenate((psi1, psi2), axis=1)


def generate_binary_sensor(m: int, dim: int) -> np.array:
    """
    Generate a binary sensor from a Bernoulli distribution
    :param m: the number of sensed data
    :param dim: the dimension/length of target signal
    :return:
        phi: a sensing matrix with size (m, dim)
    """
    return np.random.RandomState(seed=33).binomial(size=(m, dim), n=1, p=0.5)


def generate_sensor(m: int, dim: int) -> np.array:
    """
    TODO: Generate your own sensor
    :param m: the number of sensed data
    :param dim: the dimension/length of target signal
    :return:
        phi: a sensing matrix with size (m, dim)
    """
    
    poses = np.arange(0, m)
    _inv_freq = np.arange(0, dim, 2)
    _inv_freq = 1. / 10000 ** (_inv_freq / dim)
    pos_inv_freq = np.einsum("i,j->ij", poses, _inv_freq)
    emb_sin_part = np.sin(pos_inv_freq)
    emb_cos_part = np.cos(pos_inv_freq)
    phi = np.hstack((emb_sin_part, emb_cos_part))
    phi = (phi>0)*phi
    
    # ones = np.ones_like(phi)
    # phi = ones*(phi>0)+ones*(phi<0)
    # phi = (phi>0)

    # import torch as th
    # x = th.empty(1, m, dim)
    # from positional_encodings import PositionalEncoding1D
    # p_enc = PositionalEncoding1D(dim)
    # phi = p_enc(x).squeeze(0)
    # phi = (phi>0)*phi
    # phi.numpy()
    return phi


def compressive_sensing(phi: np.array, x: np.array) -> np.array:
    """
    Simulate compressive sensing
    :param phi: a sensing matrix with size (M, N), M<N
    :param x: a signal with size (N, 1)
    :return:
        y: a compressed signal with size (M, 1)
    """
    return phi @ x


def reconstruct_signal(psi: np.array, a: np.array) -> np.array:
    """
    Reconstruct a signal from a dictionary and estimated coefficient
    :param psi: a dictionary with size (N, M)
    :param a: a coefficient vector with size (M, 1)
    :return:
        x: reconstructed signal with size (N, 1)
    """
    return psi @ a


def evaluation(real: np.array, estimation: np.array, method_name: str, plot_result: bool = True) -> float:
    """
    Evaluation of the recovered signal: plot the signal and calculate the mean squared error (MSE)
    :param real: real signal with size (N, 1)
    :param estimation: estimated signal with size (N, 1)
    :param method_name: the name of the method used to recover the signal
    :param plot_result: plot result or not (You may want to plot your result when writing your report)
    :return:
        mse: mean squared error of the recovery
    """
    mse = np.sum((real - estimation) ** 2) / real.shape[0]
    result = '{}: MSE={:.4f}'.format(method_name, mse)
    print('{}\n'.format(result))
    print('{}\n'.format(result),file=open('temp','a'))
    if plot_result:
        plt.figure()
        plt.plot(real[:, 0], label='real signal')
        plt.plot(estimation[:, 0], label='recovered signal')
        plt.legend()
        plt.xlabel('n')
        plt.ylabel('x[n]')
        plt.title(result)
        plt.savefig('result_{}.pdf'.format(method_name))
        plt.close()
    return mse


# load data and pre-defined dictionary
with open('signal.pkl', 'rb') as f:
    signal = pickle.load(f)
d = signal.shape[0]
dictionary = generate_dictionary(dim=d)
n = dictionary.shape[1]
ms = [20, 40, 60, 80, 100]
plt.figure()
plt.imshow(dictionary)
plt.savefig('dictionary.pdf')
plt.close()

# TODO: If you fail to implement some of the algorithms,
#  remove the corresponding names and ensure this script can be run without errors.
method_names = ['L2reg', 'MP', 'OMP', 'BP', 'L0reg', 'ADMM']

# # """Task 1: Implement Compressive Sensing by Your Sparse Coding Algorithms"""
# # print('Task 1:\n')
# results = np.zeros((len(ms), len(method_names)))
# # generate sensing matrix and apply compressive sensing
# for i in range(len(ms)):
#     sensor = generate_binary_sensor(m=ms[i], dim=dictionary.shape[0])
#     compressed_signal = compressive_sensing(phi=sensor, x=signal)  # signal:(100,1)  compressed:(20,1)
#     for j in range(len(method_names)):
#         # baseline: recover signal with l2-regularization
#         if method_names[j] == 'L2reg':
#             a_est = l2_regularized_method(x=compressed_signal, psi=sensor @ dictionary)
#         elif method_names[j] == 'MP':
#             a_est = matching_pursuit(x=compressed_signal, psi=sensor @ dictionary, s=int(n/4)) 
#         elif method_names[j] == 'OMP':
#             a_est = orthogonal_matching_pursuit(x=compressed_signal, psi=sensor @ dictionary, s=int(n/4))     
#         elif method_names[j] == 'BP':
#             a_est = basis_pursuit(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=100000) #100000  # sensor: (20,100), dict:(100,200)
#         elif method_names[j] == 'L0reg':
#             a_est = l0_regularized_method(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=200000)  # 200000 good # sensor: (20,100), dict:(100,200)
#         elif method_names[j] == 'ADMM':
#             a_est = admm_method(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=1000) 
#         else:
#             # TODO: Replace the following line with your method, e.g., a_est = YOUR-METHOD(...)
#             a_est = np.zeros((n, 1))
#         signal_est = reconstruct_signal(psi=dictionary, a=a_est)
#         results[i, j] = evaluation(signal, signal_est,
#                                    method_name='{}+M{}'.format(method_names[j], ms[i]),
#                                    plot_result=False)


# with open('results_task1.pkl', 'wb') as f:
#     pickle.dump(results, f)

# """Task 2: Design A New Sensing Matrix and Repeat The Experiments in Task 1"""
print('Task 2:\n')
results = np.zeros((len(ms), len(method_names)))
# generate sensing matrix and apply compressive sensing
for i in range(len(ms)):
    sensor = generate_sensor(m=ms[i], dim=dictionary.shape[0])
    compressed_signal = compressive_sensing(phi=sensor, x=signal)
    for j in range(len(method_names)):
        # baseline: recover signal with l2-regularization
        if method_names[j] == 'L2reg':
            a_est = l2_regularized_method(x=compressed_signal, psi=sensor @ dictionary)
        elif method_names[j] == 'MP':
            a_est = matching_pursuit(x=compressed_signal, psi=sensor @ dictionary, s=int(n/2)) 
        elif method_names[j] == 'OMP':
            a_est = orthogonal_matching_pursuit(x=compressed_signal, psi=sensor @ dictionary, s=int(n/2))  
        elif method_names[j] == 'BP':
            a_est = basis_pursuit(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=100000) #100000  # sensor: (20,100), dict:(100,200)
        elif method_names[j] == 'L0reg':
            a_est = l0_regularized_method(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=200000)  # 200000 good # sensor: (20,100), dict:(100,200)
        elif method_names[j] == 'ADMM':
            a_est = admm_method(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=1000) 
        else:
            # TODO: Replace the following line with your method, e.g., a_est = YOUR-METHOD(...)
            a_est = np.zeros((n, 1))
        signal_est = reconstruct_signal(psi=dictionary, a=a_est)
        results[i, j] = evaluation(signal, signal_est,
                                   method_name='{}+M{}'.format(method_names[j], ms[i]),
                                   plot_result=False)

with open('results_task2.pkl', 'wb') as f:
    pickle.dump(results, f)


"""Task 3: The Robustness of Sparse Coding Methods to Noise"""
print('Task 3:\n')
sigma = [0, 0.1, 0.5, 1, 1.5]
sensor = generate_binary_sensor(m=50, dim=dictionary.shape[0])  # Rethink
results = np.zeros((len(sigma), len(sigma), len(method_names)))
for i in range(len(sigma)):
    noisy_signal = signal + sigma[i] * np.random.RandomState(seed=42).randn(signal.shape[0], 1)
    for j in range(len(sigma)):
        compressed_noisy_signal = compressive_sensing(phi=sensor, x=noisy_signal)
        compressed_noisy_signal += sigma[j] * np.random.RandomState(seed=42).randn(compressed_noisy_signal.shape[0], 1)
        for k in range(len(method_names)):
            # baseline: recover signal with l2-regularization
            if method_names[k] == 'L2reg':
                a_est = l2_regularized_method(x=compressed_noisy_signal, psi=sensor @ dictionary)
            elif method_names[k] == 'MP':
                a_est = matching_pursuit(x=compressed_noisy_signal, psi=sensor @ dictionary, s=int(n/2)) 
            elif method_names[k] == 'OMP':
                a_est = orthogonal_matching_pursuit(x=compressed_noisy_signal, psi=sensor @ dictionary, s=int(n/2)) 
            elif method_names[k] == 'BP':
                a_est = basis_pursuit(x=compressed_noisy_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=100000) #100000  # sensor: (20,100), dict:(100,200)
            elif method_names[k] == 'L0reg':
                a_est = l0_regularized_method(x=compressed_noisy_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=200000)  # 200000 good # sensor: (20,100), dict:(100,200)
            elif method_names[k] == 'ADMM':
                a_est = admm_method(x=compressed_noisy_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=1000) 
            else:
                # TODO: Replace the following line with your method, e.g., a_est = YOUR-METHOD(...)
                a_est = np.zeros((n, 1))
            signal_est = reconstruct_signal(psi=dictionary, a=a_est)
            results[i, j, k] = evaluation(signal, signal_est, method_name='{}+N{}N{}'.format(method_names[k], i, j),
                                          plot_result=False)
with open('results_task3.pkl', 'wb') as f:
    pickle.dump(results, f)


"""Task 4: Try to recover the signal based on a subset of the dictionary"""
print('Task 4:\n')
dictionary = dictionary[:, :int(n / 2)]
print('dictionary size: {}'.format(dictionary.shape))
sensor = generate_binary_sensor(m=50, dim=dictionary.shape[0])
compressed_signal = compressive_sensing(phi=sensor, x=signal)
# TODO: Implement a method to recover the signal and record its mse

results = np.zeros(len(method_names))
sensor = generate_binary_sensor(m=50, dim=dictionary.shape[0])
compressed_signal = compressive_sensing(phi=sensor, x=signal)  # signal:(100,1)  compressed:(20,1)
for i in range(len(method_names)):
    # baseline: recover signal with l2-regularization
    if method_names[i] == 'L2reg':
        a_est = l2_regularized_method(x=compressed_signal, psi=sensor @ dictionary)
    elif method_names[i] == 'MP':
        a_est = matching_pursuit(x=compressed_signal, psi=sensor @ dictionary, s=int(n/4)) 
    elif method_names[i] == 'OMP':
        a_est = orthogonal_matching_pursuit(x=compressed_signal, psi=sensor @ dictionary, s=int(n/4))     
    elif method_names[i] == 'BP':
        a_est = basis_pursuit(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=100000) #100000  # sensor: (20,100), dict:(100,200)
    elif method_names[i] == 'L0reg':
        a_est = l0_regularized_method(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=200000)  # 200000 good # sensor: (20,100), dict:(100,200)
    elif method_names[i] == 'ADMM':
        a_est = admm_method(x=compressed_signal, psi=sensor @ dictionary, gamma=0.2, max_iter=1000) 
    else:
        # TODO: Replace the following line with your method, e.g., a_est = YOUR-METHOD(...)
        a_est = np.zeros((n, 1))
    signal_est = reconstruct_signal(psi=dictionary, a=a_est)
    results[i] = evaluation(signal, signal_est,
                                method_name='{}+M{}'.format(method_names[i], 50),
                                plot_result=False)
with open('results_task4.pkl', 'wb') as f:
    pickle.dump(results, f)
