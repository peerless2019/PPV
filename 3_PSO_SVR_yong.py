from sklearn.svm import SVR
import pyswarms as ps
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import partial_dependence

def fitness_function(params, X, y, xt, yt):
    C, gamma = params
    svm_model = SVR(kernel='rbf', epsilon=0.05, C=C, gamma=gamma)
    svm_model.fit(X, y)
    y_pred = svm_model.predict(xt)
    mse = mean_squared_error(yt, y_pred)
    return mse


def optimize_svm(X, y, xt, yt, n_particles, n_iterations, dim):
    def _fitness_function(params):
        fitness_values = []
        for p in params:
            fitness = fitness_function(p, X, y, xt, yt)
            fitness_values.append(fitness)
        return fitness_values

    bounds = (np.array([0.1, 0.1]), np.array([100.0, 50.0]))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    cost_history = np.zeros(n_iterations)
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim, bounds=bounds, options=options)

    best_cost, best_params = optimizer.optimize(_fitness_function, iters=n_iterations)

    for i, cost in enumerate(optimizer.cost_history):
        cost_history[i] = cost

    C, gamma = best_params
    svm_model = SVR(kernel='rbf', epsilon=0.05, C=C, gamma=gamma)
    svm_model.fit(X, y)

    return cost_history, best_params, svm_model

if __name__ == '__main__':
    file_name = '1.xlsx'
    df = pd.read_excel(file_name)


    y = df['合成振速']
    X = df.drop('合成振速', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

    n_particles = 40
    n_iterations = 80
    dim = 2

    cost_history, best_params, svr = optimize_svm(X_train, y_train, X_test, y_test, n_particles, n_iterations, dim)
    history = svr.fit(X_train, y_train)

    y_pred_train = svr.predict(X_train)
    y_pred_test = svr.predict(X_test)

    feature_names = [f"Feature_{i}" for i in range(1, 9)]
    pdp_data = []

    for feature_idx in range(X_train.shape[1]):
        pdp_result = partial_dependence(svr, X_train, features=[feature_idx])
        feature_values = pdp_result['values'][0]
        pdp_values = pdp_result['average'][0]

        pdp_data.append({
            'Feature': feature_names[feature_idx],
            'Feature Value': feature_values,
            'PDP Value': pdp_values
        })

    df_pdp = pd.DataFrame(pdp_data)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = cost_history[-1]
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

    iterations_df = pd.DataFrame(list(range(n_iterations)))
    accuracy_df = pd.DataFrame(cost_history)

    writer = pd.ExcelWriter(r'D:\1\PSO-SVR-40.xlsx')
    y_train_df.to_excel(writer, '1', float_format='%.5f', header=None, index=False)
    y_pred_train_df.to_excel(writer, '2', float_format='%.5f', header=None, index=False)
    y_test_df.to_excel(writer, '3', float_format='%.5f', header=None, index=False)
    y_pred_test_df.to_excel(writer, '4', float_format='%.5f', header=None, index=False)
    mse_train_df.to_excel(writer, '5', float_format='%.5f', header=None, index=False)
    mse_pred_df.to_excel(writer, '6', float_format='%.5f', header=None, index=False)
    iterations_df.to_excel(writer, '7', float_format='%.5f', header=None, index=False)
    accuracy_df.to_excel(writer, '8', float_format='%.5f', header=None, index=False)
    df_pdp.to_excel(writer, '9', float_format='%.5f', header=None, index=False)
    writer._save()
