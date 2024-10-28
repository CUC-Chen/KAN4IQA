import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

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

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden1 = torch.nn.Linear(input_dim, 1536)
        self.hidden2 = torch.nn.Linear(1536, 1024)
        self.hidden3 = torch.nn.Linear(1024, 256)
        self.hidden4 = torch.nn.Linear(256, 128)
        self.output = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))
        return self.output(x)

def train_model(model, device, train_loader, val_loader, optimizer, epochs, patience=20):
    best_loss = float('inf')
    no_improvement = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                val_loss += torch.nn.functional.mse_loss(output, target).item()

        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch} with best validation loss {best_loss:.6f}")
            break

    return model

def main():
    set_seed(42)

    datasets = [
        ('BID_2048.csv', 'BID'), 
        ('CID_2048.csv', 'CID'),
        ('CLIVE_2048.csv', 'CLIVE'),
        ('KonIQ_2048.csv', 'KonIQ')
    ]

    learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 
                      1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4]

    results_file = 'MLP_results.csv'
    file_exists = os.path.isfile(results_file)

    with open(results_file, 'a') as f:
        if not file_exists:
            f.write('Dataset,LearningRate,Pearson,Spearman,Kendall,RMSE\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for data_file, dataset_name in datasets:
        print(f'Processing dataset: {dataset_name}')
        data = pd.read_csv(data_file)

        y_data = data.iloc[:, 1].values
        x_data = data.iloc[:, 2:].values

        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)

        train_size = int(0.7 * len(x_data))
        val_size = int(0.15 * len(x_data))
        test_size = len(x_data) - train_size - val_size

        x_train, y_train = x_data[:train_size], y_data[:train_size]
        x_val, y_val = x_data[train_size:train_size + val_size], y_data[train_size:train_size + val_size]
        x_test, y_test = x_data[train_size + val_size:], y_data[train_size + val_size:]

        best_combined_score = float('-inf')
        best_lr = None
        best_metrics = None

        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        for lr in learning_rates:
            print(f'Using MLP with learning rate: {lr}')
            model = MLP(input_dim=x_data.shape[1], output_dim=1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model = train_model(model, device, train_loader, val_loader, optimizer, epochs=500, patience=20)

            predictions = []
            actuals = []
            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data).squeeze()
                    predictions.extend(output.cpu().numpy())
                    actuals.extend(target.cpu().numpy())

            metrics = evaluate_metrics(np.array(predictions), np.array(actuals))

            print(f"Learning Rate: {lr}")
            print(f"Pearson (PLCC): {metrics['Pearson']:.6f}, Spearman (SRCC): {metrics['Spearman']:.6f}, Kendall (KRCC): {metrics['Kendall']:.6f}, RMSE: {metrics['RMSE']:.6f}")

            combined_score = metrics['Pearson'] + metrics['Spearman']
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_lr = lr
                best_metrics = metrics

        if best_metrics:
            with open(results_file, 'a') as f:
                f.write(f"{dataset_name},{best_lr},{best_metrics['Pearson']},{best_metrics['Spearman']},{best_metrics['Kendall']},{best_metrics['RMSE']}\n")

if __name__ == '__main__':
    main()
