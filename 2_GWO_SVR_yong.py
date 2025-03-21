
from sklearn import svm
from sklearn.svm import SVR
import numpy.random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings, pandas as pd, numpy as np, math
from sklearn.inspection import partial_dependence


def sanitized_gwo(X_train, X_test, y_train, y_test, SearchAgents_no, T, dim, lb, ub):
    Alpha_position = [0, 0]
    Beta_position = [0, 0]
    Delta_position = [0, 0]

    Alpha_score = float("inf")
    Beta_score = float("inf")
    Delta_score = float("inf")

    Positions = np.dot(rd.rand(SearchAgents_no, dim), (ub - lb)) + lb

    iterations = []
    accuracy = []

    t = 0
    while t < T:

        for i in range(0, (Positions.shape[0])):
            for j in range(0, (Positions.shape[1])):
                Flag4ub = Positions[i, j] > ub
                Flag4lb = Positions[i, j] < lb
                if Flag4ub:
                    Positions[i, j] = ub
                if Flag4lb:
                    Positions[i, j] = lb

            rbf_regressor = svm.SVR(kernel='rbf', epsilon=0.05, C=Positions[i][0],
                                    gamma=Positions[i][1]).fit(X_train, y_train)
            rbf_regressor.fit(X_train, y_train)
            y_pred = rbf_regressor.predict(X_test)
            accuracies = mean_squared_error(y_test, y_pred)


            fitness_value = accuracies * 100
            if fitness_value < Alpha_score:
                Alpha_score = fitness_value
                Alpha_position = Positions[i]
            if fitness_value > Alpha_score and fitness_value < Beta_score:
                Beta_score = fitness_value
                Beta_position = Positions[i]

            if fitness_value > Alpha_score and fitness_value > Beta_score and fitness_value < Delta_score:
                Delta_score = fitness_value
                Delta_position = Positions[i]

        a = 2 - t * (2 / T)

        for i in range(0, (Positions.shape[0])):
            for j in range(0, (Positions.shape[1])):
                r1 = rd.random(1)
                r2 = rd.random(1)
                A1 = 2 * a * r1 - a
                C1 = 0.5 + (0.5 * math.exp(-j / 500)) + (1.4 * (math.sin(j) / 30))

                D_alpha = abs(C1 * Alpha_position[j] - Positions[i, j])
                X1 = Alpha_position[j] - A1 * D_alpha

                r1 = rd.random(1)
                r2 = rd.random(1)

                A2 = 2 * a * r1 - a
                C2 = 1 + (1.4 * (1 - math.exp(-j / 500))) + (1.4 * (math.sin(j) / 30))

                D_beta = abs(C2 * Beta_position[j] - Positions[i, j])
                X2 = Beta_position[j] - A2 * D_beta

                r1 = rd.random(1)
                r2 = rd.random(1)

                A3 = 2 * a * r1 - a
                C3 = (1 / (1 + math.exp(-0.0001 * j / T))) + (
                            (0.5 - 2.5) * ((j / T) ** 2))

                D_delta = abs(C3 * Delta_position[j] - Positions[i, j])
                X3 = Delta_position[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3


        t = t + 1
        iterations.append(t)
        accuracy.append(abs(Alpha_score / 100))

    best_C = Alpha_position[0]
    best_gamma = Alpha_position[1]

    return best_C, best_gamma, iterations, accuracy

if __name__ == '__main__':

    file_name = '1.xlsx'
    df = pd.read_excel(file_name)


    y = df['合成振速']
    X = df.drop('合成振速', axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

    SearchAgents_no = 80
    T = 80
    dim = 2
    lb = 0.01
    ub = 100

    best_C, best_gamma, iterations, accuracy = sanitized_gwo(X_train, X_test, y_train, y_test, SearchAgents_no, T, dim,
                                                             lb, ub)



    svr_regressor = SVR(kernel='rbf', epsilon=0.05, C=best_C, gamma=best_gamma)
    history = svr_regressor.fit(X_train, y_train)

    y_pred_train = svr_regressor.predict(X_train)
    y_pred_test = svr_regressor.predict(X_test)

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

    iterations_df = pd.DataFrame(iterations)
    accuracy_df = pd.DataFrame(accuracy)

    writer = pd.ExcelWriter(r'D:\1\GWO-SVR-80.xlsx')
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