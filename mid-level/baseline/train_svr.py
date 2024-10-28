import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau
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
        ('BID_x_15_BID.csv', 'BID_y_y_real_BID.csv', 'BID'),
        ('CID_x_15_CID.csv', 'CID_y_y_real_CID2013.csv', 'CID'),
        ('CLIVE_x_15_CLIVE.csv', 'CLIVE_y_y_real_CLIVE.csv', 'CLIVE'),
        ('KonIQ_x_15_KonIQ.csv', 'KonIQ_y_y_real_KONIQ.csv', 'KonIQ')
    ]

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results_file = 'SVR_results.csv'
    file_exists = os.path.isfile(results_file)

    with open(results_file, 'a') as f:
        if not file_exists:
            f.write('Dataset,Kernel,Pearson,Spearman,Kendall,RMSE\n')

    for x_file, y_file, dataset_name in datasets:
        print(f'Processing dataset: {dataset_name}')
        x_data = pd.read_csv(x_file).values
        y_data = pd.read_csv(y_file).values.squeeze()

        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)

        train_size = int(0.8 * len(x_data))
        x_train, y_train = x_data[:train_size], y_data[:train_size]
        x_test, y_test = x_data[train_size:], y_data[train_size:]

        for kernel in kernels:
            print(f'Using SVR with kernel: {kernel}')
            model = SVR(kernel=kernel, C=10, epsilon=0.1, gamma='scale')

            model.fit(x_train, y_train)

            test_predictions = model.predict(x_test)
            metrics = evaluate_metrics(test_predictions, y_test)

            print(f"Kernel: {kernel}")
            print(f"Pearson (PLCC): {metrics['Pearson']:.6f}, Spearman (SRCC): {metrics['Spearman']:.6f}, Kendall (KRCC): {metrics['Kendall']:.6f}, RMSE: {metrics['RMSE']:.6f}")

            with open(results_file, 'a') as f:
                f.write(f"{dataset_name},{kernel},{metrics['Pearson']},{metrics['Spearman']},{metrics['Kendall']},{metrics['RMSE']}\n")

if __name__ == '__main__':
    main()
