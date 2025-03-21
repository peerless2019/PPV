
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HPO(object):
    # 定义初始化方法
    def __init__(self, m, T, lb, ub, R, C, X_train, y_train, X_test, y_test):
        self.M = m
        self.T = T
        self.lb = lb
        self.ub = ub
        self.R = R
        self.C = C
        self.b = 0.1

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def init_x(self):
        x = np.random.uniform(self.lb, self.ub, (self.M, self.R, self.C))
        return x


    def fitness(self, x):

        if abs(x[0][0][1]) > 0:
            gamma = abs(x[0][0][1])
        else:
            gamma = 'scale'

        if abs(x[0][0][0]) > 0:
            C = abs(x[0][0][0])
        else:
            C = 1.0

        svr = SVR(kernel='rbf', epsilon=0.05, C=C, gamma=gamma).fit(X_train, y_train)

        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        accuracies = mean_squared_error(y_test, y_pred)
        fitness_value = accuracies * np.ones(m)

        return fitness_value


    def main(self):
        x = self.init_x()
        fitness = self.fitness(x)
        fitness_best = fitness.min()
        x_num = list(fitness).index(fitness_best)
        x_best = x[x_num]

        fitness_best_list = []
        fitness_best_list.append(fitness_best)

        iterations = []

        for it in range(self.T):
            C = 1 - it * (0.98 / self.T)
            R1 = np.random.choice([0, 1], (self.M, self.R, self.C))
            R3 = np.random.choice([0, 1], (self.M, self.R, self.C))
            R2 = np.random.rand(self.M)

            R1 = R1 - C
            R1[R1 <= 0] = 0
            R1[R1 > 0] = 1
            R_IDX = R1.copy()
            R_IDX = 1 - R_IDX
            z = np.zeros((self.M, self.R, self.C))
            for i in range(self.M):
                z[i] = R2[i] * R_IDX[i]
            z += R3 * R1

            R5 = np.random.rand()
            x_new = np.zeros((self.M, self.R, self.C))
            if R5 < self.b:
                # u
                u = np.zeros((self.R, self.C))
                for i in range(self.M):
                    u += x[i]
                u = u / self.M

                Deuc_par = x - u
                Deuc = np.zeros(self.M)
                for i in range(self.M):
                    Deuc[i] = np.sqrt(np.sum(Deuc_par[i] ** 2))
                kbest = round(self.M * C)
                Deuc_max = np.max(sorted((Deuc)[:kbest]))
                Deuc = Deuc[:kbest]
                num_par = np.where(Deuc == Deuc_max)
                num = num_par[0][0]
                P_pos = x[num]
                x_new = x + 0.5 * (2 * C * z * P_pos - x + 2 * (1 - C) * z * u - x)
            else:
                R4 = np.random.rand()
                for i in range(self.M):
                    tmp_tmp = C * z * np.cos(2 * np.pi * R4) * (x_best - x[i])
                    x_new[i] = x_best + tmp_tmp[0]
            x = x_new
            fitness = self.fitness(x)
            fitness_best_new = fitness.min()

            if fitness_best_new <= fitness_best:
                fitness_best = fitness_best_new
                x_num = list(fitness).index(fitness_best)
                x_best = x[x_num]
            else:
                fitness_best = fitness_best
            fitness_best_list.append(fitness_best)
            iterations.append(it)

        return x_best, fitness_best, fitness_best_list, iterations  # 返回数据


if __name__ == '__main__':

    df = pd.read_excel('1.xlsx')


    y = df['合成振速']
    X = df.drop('合成振速', axis=1)  # 特征


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

    m = 20
    T = 80
    lb = -10
    ub = 10
    R = 1
    C = 2

    hpo = HPO(m=m, T=T, lb=lb, ub=ub, R=R, C=C, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    Best_pos, Best_score, accuracy, iterations = hpo.main()
    if abs(Best_pos[0][0]) > 0:
        best_C = abs(Best_pos[0][0]) * 10
    else:
        best_C = abs(Best_pos[0][0]) + 1

    if abs(Best_pos[0][1]) > 0:
        best_gamma = abs(Best_pos[0][1]) * 10
    else:
        best_gamma = (abs(Best_pos[0][1]) + 0.1) * 10

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

    iterations_df = pd.DataFrame(iterations)
    accuracy_df = pd.DataFrame(accuracy)

    writer = pd.ExcelWriter(r'D:\1\HPO-SVR-20.xlsx')
    y_train_df.to_excel(writer, '1', float_format='%.5f', header=None, index=False)
    y_pred_train_df.to_excel(writer, '2', float_format='%.5f', header=None, index=False)
    y_test_df.to_excel(writer, '3', float_format='%.5f', header=None, index=False)
    y_pred_test_df.to_excel(writer, '4', float_format='%.5f', header=None, index=False)
    mse_train_df.to_excel(writer, '5', float_format='%.5f', header=None, index=False)
    mse_pred_df.to_excel(writer, '6', float_format='%.5f', header=None, index=False)
    iterations_df.to_excel(writer, '7', float_format='%.5f', header=None, index=False)
    accuracy_df.to_excel(writer, '8', float_format='%.5f', header=None, index=False)
    writer._save()
