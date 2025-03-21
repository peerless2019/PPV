from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import warnings, pandas as pd, numpy as np
from sklearn.inspection import partial_dependence
def import_data():

    df = pd.read_excel('1.xlsx')

    y = df['合成振速']
    X = df.drop('合成振速', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

    return X_train, X_test, y_train, y_test

def fitness_spaction(parameter):

    X_train, X_test, y_train, y_test = import_data()
    c = parameter[0]
    g = parameter[1]
    clf = SVR(kernel='rbf', epsilon=0.05, C=c, gamma=g)
    clf.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies = mean_squared_error(y_test, y_pred)
    return (accuracies)


def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]

    return temp

def SSA(pop, M, c, d, dim, fun):

    P_percent = 0.2
    pNum = round(pop * P_percent)
    lb = c * np.ones((1, dim))
    ub = d * np.ones((1, dim))
    X = np.zeros((pop, dim))
    fit = np.zeros((pop, 1))

    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
        fit[i, 0] = fun(X[i, :])
    pFit = fit
    pX = X
    fMin = np.min(fit[:, 0])
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :]
    Convergence_curve = np.zeros((1, M))
    # 进行迭代
    for t in range(M):
        sortIndex = np.argsort(pFit.T)
        fmax = np.max(pFit[:, 0])
        B = np.argmax(pFit[:, 0])
        worse = X[B, :]
        r2 = np.random.rand(1)
        if r2 < 0.8:
            for i in range(pNum):
                r1 = np.random.rand(1)
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] * np.exp(-(i) / (r1 * M))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        elif r2 >= 0.8:

            for i in range(pNum):
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] + np.random.rand(1) * np.ones((1, dim))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        bestII = np.argmin(fit[:, 0])
        bestXX = X[bestII, :]
        for ii in range(pop - pNum):
            i = ii + pNum
            A = np.floor(np.random.rand(1, dim) * 2) * 2 - 1
            if i > pop / 2:
                X[sortIndex[0, i], :] = np.random.rand(1) * np.exp(worse - pX[sortIndex[0, i], :] / np.square(i))
            else:
                X[sortIndex[0, i], :] = bestXX + np.dot(np.abs(pX[sortIndex[0, i], :] - bestXX),
                                                        1 / (A.T * np.dot(A, A.T))) * np.ones((1, dim))
            X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
            fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        arrc = np.arange(len(sortIndex[0, :]))


        c = np.random.permutation(arrc)
        b = sortIndex[0, c[0:20]]
        for j in range(len(b)):
            if pFit[sortIndex[0, b[j]], 0] > fMin:
                X[sortIndex[0, b[j]], :] = bestX + np.random.rand(1, dim) * np.abs(
                    pX[sortIndex[0, b[j]], :] - bestX)
            else:
                X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :] + (2 * np.random.rand(1) - 1) * np.abs(
                    pX[sortIndex[0, b[j]], :] - worse) / (pFit[sortIndex[0, b[j]]] - fmax + 10 ** (-50))
            X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)
            fit[sortIndex[0, b[j]], 0] = fun(X[sortIndex[0, b[j]]])
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :]
        Convergence_curve[0, t] = fMin
    return fMin, bestX, Convergence_curve


if __name__ == "__main__":

    data = pd.read_excel('1.xlsx')


    X_train, X_test, y_train, y_test = import_data()

    SearchAgents_no = 60
    Max_iteration = 80
    dim = 2
    lb = [0.01, 0.01]
    ub = [1000, 1000]

    fMin, bestX, curve = SSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction)
    SSA_curve = curve.T

    clf = SVR(kernel='rbf', epsilon=0.05, C=bestX[0], gamma=bestX[1])
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = SSA_curve[-1]
    r2_test = r2_score(y_test, y_pred_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    y_pred_train_df = pd.DataFrame(y_pred_train)
    y_pred_test_df = pd.DataFrame(y_pred_test)

    r2_train = np.array([r2_train])
    r2_test = np.array([r2_test])
    r2_train_df = pd.DataFrame(r2_train)
    r2_test_df = pd.DataFrame(r2_test)

    mse_train = np.array([mse_train])
    mse_test = np.array([mse_test])
    mse_train_df = pd.DataFrame(mse_train)
    mse_pred_df = pd.DataFrame(mse_test)

    mae_train = np.array([mae_train])
    mae_test = np.array([mae_test])
    mae_train_df = pd.DataFrame(mae_train)
    mae_pred_df = pd.DataFrame(mae_test)

    iterations_df = pd.DataFrame(range(Max_iteration))
    accuracy_df = pd.DataFrame(SSA_curve)

    writer = pd.ExcelWriter(r'D:\1\SSA-SVR-60.xlsx')
    y_train_df.to_excel(writer, '1', float_format='%.5f', header=None, index=False)
    y_pred_train_df.to_excel(writer, '2', float_format='%.5f', header=None, index=False)
    y_test_df.to_excel(writer, '3', float_format='%.5f', header=None, index=False)
    y_pred_test_df.to_excel(writer, '4', float_format='%.5f', header=None, index=False)
    mse_train_df.to_excel(writer, '5', float_format='%.5f', header=None, index=False)
    mse_pred_df.to_excel(writer, '6', float_format='%.5f', header=None, index=False)
    mae_train_df.to_excel(writer, '7', float_format='%.5f', header=None, index=False)
    mae_pred_df.to_excel(writer, '8', float_format='%.5f', header=None, index=False)
    iterations_df.to_excel(writer, '9', float_format='%.5f', header=None, index=False)
    accuracy_df.to_excel(writer, '10', float_format='%.5f', header=None, index=False)
    r2_test_df.to_excel(writer, '11', float_format='%.5f', header=None, index=False)
    r2_train_df.to_excel(writer, '12', float_format='%.5f', header=None, index=False)
    writer._save()

