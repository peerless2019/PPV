
import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()

    return X



def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin



def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def levy_distribution(beta, dim):

    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)

    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)

    step = u / abs(v) ** (1 / beta)
    LF = 0.01 * step

    return LF



def error_rate(X_train, y_train, X_test, y_test, x):
    if abs(x[0]) > 0:
        gamma = abs(x[0]) / 10
    else:
        gamma = 'scale'

    if abs(x[1]) > 0:
        C = abs(x[1]) / 10
    else:
        C = 1.0


    svr = SVR(kernel='rbf', epsilon=0.05, C=C, gamma=gamma).fit(X_train, y_train)

    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    accuracies = mean_squared_error(y_test, y_pred)

    return accuracies



def Fun(X_train, y_train, X_test, y_test, x, opts):

    alpha = 0.5
    beta = 1 - alpha

    max_feat = len(x)

    num_feat = np.sum(x == 1)

    error = error_rate(X_train, y_train, X_test, y_test, x, opts)

    cost = (alpha * error + beta * (num_feat / max_feat))

    return cost



def jfs(X_train, y_train, X_test, y_test, opts):

    ub = 8
    lb = -8
    thres = 0
    beta = 1.5

    N = opts['N']
    max_iter = opts['T']
    if 'beta' in opts:
        beta = opts['beta']


    dim = np.size(X_train, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')


    X = init_position(lb, ub, N, dim)

    fit = np.zeros([N, 1], dtype='float')
    Xrb = np.zeros([1, dim], dtype='float')
    fitR = float('inf')

    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:

        Xbin = binary_conversion(X, thres, N, dim)

        for i in range(N):
            fit[i, 0] = Fun(X_train, y_train, X_test, y_test, Xbin[i, :], opts)
            if fit[i, 0] < fitR:
                Xrb[0, :] = X[i, :]
                fitR = fit[i, 0]

        curve[0, t] = fitR.copy()

        t += 1

        X_mu = np.zeros([1, dim], dtype='float')
        X_mu[0, :] = np.mean(X, axis=0)

        for i in range(N):
            E0 = -1 + 2 * rand()
            E = 2 * E0 * (1 - (t / max_iter))

            if abs(E) >= 1:
                q = rand()
                if q >= 0.5:
                    k = np.random.randint(low=0, high=N)
                    r1 = rand()
                    r2 = rand()
                    for d in range(dim):
                        X[i, d] = X[k, d] - r1 * abs(X[k, d] - 2 * r2 * X[i, d])
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                elif q < 0.5:
                    r3 = rand()
                    r4 = rand()
                    for d in range(dim):
                        X[i, d] = (Xrb[0, d] - X_mu[0, d]) - r3 * (lb[0, d] + r4 * (ub[0, d] - lb[0, d]))
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])


            elif abs(E) < 1:
                J = 2 * (1 - rand())
                r = rand()

                if r >= 0.5 and abs(E) >= 0.5:
                    for d in range(dim):
                        DX = Xrb[0, d] - X[i, d]
                        X[i, d] = DX - E * abs(J * Xrb[0, d] - X[i, d])
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                elif r >= 0.5 and abs(E) < 0.5:
                    for d in range(dim):
                        DX = Xrb[0, d] - X[i, d]
                        X[i, d] = Xrb[0, d] - E * abs(DX)
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                elif r < 0.5 and abs(E) >= 0.2:
                    LF = levy_distribution(beta, dim)
                    Y = np.zeros([1, dim], dtype='float')
                    Z = np.zeros([1, dim], dtype='float')

                    for d in range(dim):

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X[i, d])

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])

                    for d in range(dim):

                        Z[0, d] = Y[0, d] + rand() * LF[d]

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])


                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)

                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY
                        X[i, :] = Y[0, :]
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ
                        X[i, :] = Z[0, :]

                elif r < 0.5 and abs(E) < 0.5:

                    LF = levy_distribution(beta, dim)
                    Y = np.zeros([1, dim], dtype='float')
                    Z = np.zeros([1, dim], dtype='float')

                    for d in range(dim):

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X_mu[0, d])

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])

                    for d in range(dim):

                        Z[0, d] = Y[0, d] + rand() * LF[d]

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])

                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)

                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)

                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY
                        X[i, :] = Y[0, :]
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ
                        X[i, :] = Z[0, :]
        ss = curve.tolist()
        ss = np.array(ss).reshape(max_iter)
    return X, ss


if __name__ == '__main__':

    file_name = '1.xlsx'
    df = pd.read_excel(file_name)

    y = df['合成振速']
    X = df.drop('合成振速', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

    N = 60
    T = 80

    opts = {'N': N, 'T': T}


    fmdl, accuracy = jfs(X_train, y_train, X_test, y_test, opts)

    if abs(fmdl[0][0]) > 0:
        best_C = abs(fmdl[0][0])
    else:
        best_C = abs(fmdl[0][0]) + 1

    if abs(fmdl[0][1]) > 0:
        best_gamma = abs(fmdl[0][1])
    else:
        best_gamma = (abs(fmdl[0][1]) + 0.1)

    svr = SVR(kernel='rbf', epsilon=0.05, C=best_C, gamma=best_gamma)
    svr.fit(X_train, y_train)

    y_pred_train = svr.predict(X_train)
    y_pred_test = svr.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = accuracy[-1]
    r2_test = r2_score(y_test, y_pred_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    y_pred_train_df = pd.DataFrame(y_pred_train)
    y_pred_test_df = pd.DataFrame(y_pred_test)

    mse_train = np.array([mse_train])
    mse_test = np.array([mse_test])
    mse_train_df = pd.DataFrame(mse_train)
    mse_pred_df = pd.DataFrame(mse_test)

    iterations_df = pd.DataFrame(range(T))
    accuracy_df = pd.DataFrame(accuracy)

    writer = pd.ExcelWriter(r'D:\1\HHO-SVR-20.xlsx')
    y_train_df.to_excel(writer, '1', float_format='%.5f', header=None, index=False)
    y_pred_train_df.to_excel(writer, '2', float_format='%.5f', header=None, index=False)
    y_test_df.to_excel(writer, '3', float_format='%.5f', header=None, index=False)
    y_pred_test_df.to_excel(writer, '4', float_format='%.5f', header=None, index=False)
    mse_train_df.to_excel(writer, '5', float_format='%.5f', header=None, index=False)
    iterations_df.to_excel(writer, '7', float_format='%.5f', header=None, index=False)
    accuracy_df.to_excel(writer, '8', float_format='%.5f', header=None, index=False)
    writer._save()
