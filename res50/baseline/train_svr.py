import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

def set_seed(seed):
    np.random.seed(seed)

def evaluate_metrics(predictions, actuals):
    pearson_corr, _ = pearsonr(predictions, actuals)
    spearman_corr, _ = spearmanr(predictions, actuals)
    kendall_corr, _ = kendalltau(predictions, actuals)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return {
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "Kendall": kendall_corr,
        "RMSE": rmse
    }

def main():
    set_seed(42)
    datasets = [
        ('BID_2048.csv', 'BID'), 
        ('CID_2048.csv', 'CID'),
        ('CLIVE_2048.csv', 'CLIVE'),
        ('KonIQ_2048.csv', 'KonIQ')
    ]
    kernels = ['rbf']
    results_file = 'SVR_results_sklearn.csv'
    file_exists = os.path.isfile(results_file)

    with open(results_file, 'a') as f:
        if not file_exists:
            f.write('Dataset,Kernel,Pearson,Spearman,Kendall,RMSE\n')

    for data_file, dataset_name in datasets:
        print(f'Processing dataset: {dataset_name}')

        data = pd.read_csv(data_file)

        y_data = data.iloc[:, 1].values
        x_data = data.iloc[:, 2:].values

        scaler = StandardScaler()
        train_size = int(0.8 * len(x_data))

        x_train = scaler.fit_transform(x_data[:train_size])
        x_test = scaler.transform(x_data[train_size:])
        y_train = y_data[:train_size]
        y_test = y_data[train_size:]

        for kernel in kernels:
            print(f'Using SVR with kernel: {kernel}')

            model = SVR(kernel=kernel, C=10, epsilon=0.1, gamma='scale')

            model.fit(x_train, y_train)

            predictions = model.predict(x_test)

            metrics = evaluate_metrics(predictions, y_test)
            print(f"Kernel: {kernel}")
            print(f"Pearson (PLCC): {metrics['Pearson']:.6f}, Spearman (SRCC): {metrics['Spearman']:.6f}, Kendall (KRCC): {metrics['Kendall']:.6f}, RMSE: {metrics['RMSE']:.6f}")

            with open(results_file, 'a') as f:
                f.write(f"{dataset_name},{kernel},{metrics['Pearson']},{metrics['Spearman']},{metrics['Kendall']},{metrics['RMSE']}\n")

    print("Results saved to SVR_results_sklearn.csv")

if __name__ == '__main__':
    main()
