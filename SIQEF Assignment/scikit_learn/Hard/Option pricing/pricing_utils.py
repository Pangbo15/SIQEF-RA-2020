import numpy as np
import pandas as pd
from numpy.random import standard_normal, seed, uniform, randint
from scipy.stats import norm
from fangoosterlee import Heston, HestonParam, cosmethod


def GBM_multi_T(N, T, S_0, r, sigma, Sigma):
    """
    Compute d-asset prices paths at time T starting at time 0 with price S_0 following GBM
    Parameters
    :param N: nb of paths
    :param T: time at which to compute price
    :param S_0: init price
    :param r: risk-free rate
    :param sigma: vector of volatility of each asset
    :param Sigma: matrix of covariances
    :return: two dimensional array, matrix of prices (dxN)
    """
    d = S_0.shape[0]
    S_T = np.exp(T * np.dot(np.diag(r - np.square(sigma)/2), np.ones((d, N)))
             + np.sqrt(T) * np.dot(Sigma, np.random.randn(d, N)))
    S_T = np.dot(np.diag(S_0), S_T)
    return S_T


def GBM_multi(N, T, M, S_0, r, sigma, Sigma):
    """
    multi-dimensional GBM path generation starting S_0 with maturity T
     and M time step
    :param N: nb of paths
    :param T: time at which to compute price
    :param M: number of time step
    :param S_0: init prices array
    :param r: risk-free rate
    :param sigma: vector of volatility of each asset
    :param Sigma: matrix of covariances
    :return: nd-array of size (d x N x M)
    """
    delta_t = T/M
    d = S_0.shape[0]
    S = np.zeros((d, N, M))
    S[:,:,0] = np.tile(S_0, (N, 1)).T
    for m in range(1, M):
        S[:,:,m] = S[:,:,m-1] * np.exp(delta_t * np.dot(np.diag(r - np.square(sigma)/2), np.ones((d, N)))
             + np.sqrt(delta_t) * np.dot( Sigma , np.random.randn(d, N)))
    return S


def bs_put(S0, K, r, sigma, tau):
    """BS Closed formula for European call option"""
    d1 = (np.log(S0 / K) + (r + 1 / 2 * sigma ** 2) * tau) / sigma / np.sqrt(tau)
    d2 = (np.log(S0 / K) + (r - 1 / 2 * sigma ** 2) * tau) / sigma / np.sqrt(tau)
    price = K * np.exp(-r * tau) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price


def bs_call(S0, K, r, sigma, tau):
    """BS Closed formula for European put option"""
    d1 = (np.log(S0 / K) + (r + 1 / 2 * sigma ** 2) * tau) / sigma / np.sqrt(tau)
    d2 = (np.log(S0 / K) + (r - 1 / 2 * sigma ** 2) * tau) / sigma / np.sqrt(tau)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return price


def bs_vol_imp_call(S, K, T, V, r, sigma_init, tol=1e-7):
    """
    Newton R. root finding method for implicit volatility
    :param S: spot price
    :param K: strike price
    :param T: time to maturity
    :param V: Call value
    :param r: risk-free rate
    :param sigma_init:  initial guess of volatility
    :param tol:
    :return:
    """
    d1 = (np.log(S / K) + (r - 0.5 * sigma_init ** 2) * T) / (sigma_init * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma_init ** 2) * T) / (sigma_init * np.sqrt(T))
    fx = S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0) - V
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    x0 = sigma_init
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tol:
        xold = xnew
        xnew = (xnew - fx - V) / vega

    return abs(xnew)


def payoff_call_basket(a, S_T, K):
    """
    payoff call basket option
    :param a (array-like): weight vector of dim d
    :param S_T (array-like): price matrix of dim dxN
    :param K (float): strike
    :return: array of dim N
    """
    I_T = np.dot(np.transpose(a), S_T)
    G = np.maximum(I_T - K, 0)
    return G


def payoff_put_basket(a, S_T, K):
    """
    payoff put basket option
    :param a (array-like): weight vector of dim d
    :param S_T (array-like): price matrix of dim dxN
    :param K (float): strike
    :return: array of dim N
    """
    I_T = np.dot(a, S_T)
    G = np.maximum(K - I_T, 0)
    return G


def payoff_worstof_put_basket(S_T,K):
    """
    payoff put basket of kind max(K - max(S1,..,Sd))
    :param a (array-like): weight vector of dim d
    :param S_T (array-like): price matrix of dim dxN
    :param K (float): strike
    :return: array of dim N
    """
    I_T = S_T.max(axis=0)
    G   = np.maximum(K - I_T, 0)
    return G


def basket_mc_price(N, S_0, K, r, T, sigma, Sigma, a, fonction_payoff):
    """
    Pricer for basket (multi-asset) European-type options in BS framework
    :param N: nb of paths
    :param S_0:  init price
    :param K: number of time step
    :param r: risk-free rate
    :param T: time at which to compute price
    :param sigma: vector of volatility of each asset
    :param Sigma: matrix of covariances
    :param a: weight
    :param fonction_payoff:
    :return: price (tuple) : mean, var, mc error
    """
    S_T = GBM_multi_T(N, T, S_0, r, sigma, Sigma)

    payoff = np.exp(-r * T) * fonction_payoff(a, S_T, K)
    est_price = np.mean(payoff)
    standard_dev = np.std(payoff)
    err = 1.96 * standard_dev / np.sqrt(N)

    return est_price, standard_dev, err


def cos_heston_price(S_0, K, r, T, v, v_0, kappa, gamma, rho, call=True):
    """
    Pricer for European option (single asset) under Heston dynamic
    :param S_0: init stock price
    :param K: strike
    :param r: risk-free rate
    :param T: maturity
    :param v: long term mean variance
    :param v_0: initial variance
    :param kappa: reversion speed
    :param gamma: vol coefficient of vol
    :param rho: correlation between stock and volatility dynamic
    :param call: call if True, put otherwise
    :return: Option price
    """
    moneyness = np.log(K / S_0) - r * T
    param = HestonParam(lm=kappa, mu=v, eta=gamma, rho=rho, sigma=v_0)
    model = Heston(param, r, T)
    P = cosmethod(model, moneyness=moneyness, call=call)
    return P[0]


def basis_function_2D(R1, R2, X, Y, basis='laguerre'):
    """
    basis_function_laguerre - Create Laguerre polynomial for basis of two asset path
    American option pricing purpose

    :param R1:
    :param R2:
    :param X:
    :param Y:
    :param basis:
    :return:
    """
    if R1 == 0:
        return 0. * X
    if R2 == 0:
        return 0. * X
    else:
        if basis == 'laguerre':
            coef = np.zeros((R1, R2))
            coef[-1, -1] = 1
            value = np.polynomial.laguerre.lagval2d(X, Y, coef)
            return value

        if basis == 'monomial':
            return np.multiply(np.power(X, R1), np.power(Y, R2))


def function_A_mat_2D(t, N, P, R, basis='laguerre', reg_param=1e-8):
    """compute E[psi_r(Xi)psi_s(Xi)] in least square formula, see Reference"""
    B_psi_psi = np.zeros((R ** 2, R ** 2))

    for i in range(R):
        for j in range(R):
            for k in range(R):
                for l in range(R):
                    psi_1 = basis_function_2D(i + 1, j + 1, P[0, :, t], P[1, :, t], basis=basis)
                    psi_2 = basis_function_2D(k + 1, l + 1, P[0, :, t], P[1, :, t], basis=basis)

                    B_psi_psi[(i) * (j), (k) * (l)] = sum(np.multiply(psi_1, psi_2))
                    B_psi_psi[(i) * (j), (k) * (l)] = B_psi_psi[(i) * (j), (k) * (l)] / N

    B_psi_psi = B_psi_psi + reg_param * np.eye(R ** 2)

    return B_psi_psi


def function_B_vec_2D(t, N, R, P, V, basis='laguerre'):
    """compute E[V(X_i+1)psi_r(Xi)] in least square
     minimization result formula """
    B_V_psi = np.zeros((R ** 2))

    for i in range(R):
        for j in range(R):
            B_V_psi[(i) * (j)] = sum(
                np.multiply(V, basis_function_2D(i + 1, j + 1, P[0, :, t], P[1, :, t], basis=basis)))
            B_V_psi[(i) * (j)] = B_V_psi[(i) * (j)] / N

    return B_V_psi


def continuation_val_2D(t, R, P, Beta, N, basis='laguerre'):
    """compute estimator of continuation value given regression coef"""
    cont_val = np.zeros(N)
    for i in range(R):
        for j in range(R):
            cont_val = cont_val + basis_function_2D(i + 1, j + 1, P[0, :, t], P[1, :, t], basis=basis) * \
                              Beta[t, (i) * (j)]
    return cont_val


def LS_backward_2D(P, payoff_fun, K, R=4, discount_factor=1, basis='laguerre'):
    """
    Learning phase of the Least-square MC algo

    :param P: multi-dimensional Prices paths array (2 x N x M)
    :param payoff_fun: callable, payoff fun used to
    :param K:  float, strike used
    :param R: int,  order of polynomial basis
    :param discount_factor: float, discount factor
    :param basis:
    :return:
    Beta: matrix of coef of each time step regression (size M x R^2)
    npv: matrix of cashflow (size M x N)
    """
    assert P.shape[0] == 2, 'price matrix should have size 2 x N x M'
    N = P.shape[1]  # nb of path
    M = P.shape[2]  # nb of time step
    V = payoff_fun(P[:, :, M - 1], K)
    Beta = np.zeros((M, R ** 2))
    npv = []
    for t in range(M - 1, -1, -1):
        A_mat = function_A_mat_2D(t, N, P, R, basis=basis)
        B_vec = function_B_vec_2D(t, N, R, P, V, basis=basis)
        Beta[t, :] = np.dot(np.linalg.inv(A_mat), B_vec) # fit the regression coef
        cont_val = continuation_val_2D(t, R, P, Beta, N, basis=basis)
        V = np.where(cont_val < payoff_fun(P[:, :, t], K), payoff_fun(P[:, :, t], K), discount_factor * V)
        npv.append(V.copy())
    npv = np.array(npv)
    return Beta, npv


def LS_forward_2D(P, Beta, payoff_fun, K, discount_factor=1, basis='laguerre'):
    """forward evaluation for pricing using coef of fitted regressions
        Returns
        -------
        mean, variance and err of MC
    """
    R = int(np.sqrt(Beta.shape[1]))
    M = P.shape[2]
    N = P.shape[1]
    V = discount_factor ** M * payoff_fun(P[:, :, M - 1], K)
    cont_val = np.zeros((N, M))
    for t in range(M):
        cont_val[:, t] = continuation_val_2D(t, R, P, Beta, N, basis=basis)
    for n in range(N):
        for t in range(M):
            if cont_val[n, t] < payoff_fun(P[:, :, t], K)[n]:
                V[n] = discount_factor ** t * payoff_fun(P[:, :, t], K)[n]
                break
    mean = np.sum(V) / N
    var = np.sum(V - mean) ** 2 / N
    err = np.sqrt(var / N)
    return mean, var, err


def LSMC_2D_pricer(N, R, T, M, S_0, K, r, sigma, Sigma, payoff_fun, basis='laguerre'):
    """

    :param N: nb of MC paths
    :param R: polynomial order for basis
    :param T: maturity
    :param M: nb of exercise date
    :param S_0: init price array, must be lenght 2
    :param K: strike value
    :param r: risk-free rate
    :param sigma: array of asset vol
    :param Sigma: asset covariance matrix
    :param payoff_fun: fun for payoff
    :param basis: kind of basis ('laguerre', 'monomial')
    :return:
        - tuple (mean, var, err):  price
        - npv: matrix of cashflow (M x N) of learning phase
        - P_train: matrix of prices (2 x N x M) used for learning phase
    """
    np.random.seed(42)
    P_train = GBM_multi(N, T, M, S_0, r, sigma, Sigma)
    delta_t = T / M
    discount_factor = np.exp(-r * delta_t)
    Beta, npv = LS_backward_2D(P_train, payoff_fun, K, R, discount_factor, basis=basis)
    P_test = GBM_multi(N, T, M, S_0, r, sigma, Sigma) # simulate new set of prices
    np.random.seed(100)
    return LS_forward_2D(P_test, Beta, payoff_fun, K, discount_factor, basis=basis), npv, P_train


