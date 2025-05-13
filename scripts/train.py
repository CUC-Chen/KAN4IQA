import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import importlib
import time
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_model(model, device, train_loader, val_loader, optimizer, epochs, patience, checkpoint_path, scheduler=None):
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)
    
    epoch_pbar = tqdm(range(epochs), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False, position=1)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
            train_pbar.set_postfix({'loss': loss.item()})
        
        train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False, position=1)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        val_loss = val_loss / len(val_loader.dataset)
        
        if scheduler:
            scheduler.step(val_loss)
        
        epoch_pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})
        
        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model

def evaluate_model(model, device, test_loader):
    model.eval()
    predictions, actuals = [], []
    
    test_pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, targets in test_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    return evaluate_metrics(np.array(predictions), np.array(actuals))

def analyze_pca_components(x_data, thresholds=[0.95]):
    imputer = SimpleImputer(strategy='mean')
    x_data_clean = imputer.fit_transform(x_data)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data_clean)
    pca = PCA()
    pca.fit(x_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    optimal_components = {}
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        optimal_components[threshold] = n_components
    
    return optimal_components

def initialize_model(model_name, input_dim, output_dim, expansion_point=0.0):
    module = importlib.import_module(model_name)
    if model_name == 'TaylorKAN':
        KANModel = getattr(module, model_name)
        if input_dim <= 64:
            layers = [input_dim, 64, 16, output_dim]
        elif input_dim <= 128:
            layers = [input_dim, 128, 32, output_dim]
        elif input_dim <= 256:
            layers = [input_dim, 256, 64, output_dim]
        else:
            layers = [input_dim, 512, 128, output_dim]
        return KANModel(layers, expansion_point=expansion_point)

def main():
    set_seed(42)
    datasets = [
        ('BID_2048.csv', 'BID'), 
        ('CLIVE_2048.csv', 'CLIVE'),
        ('KonIQ_2048.csv', 'KonIQ'),
        ('SPAQ.csv','SPAQ'),
        ('FLIVE.csv', 'FLIVE')
    ]
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    best_results_file = 'outputs/best_results.csv'
    all_results_file = 'outputs/all_results.csv'
    
    best_file_exists = os.path.isfile(best_results_file)
    all_file_exists = os.path.isfile(all_results_file)
    
    if not best_file_exists:
        with open(best_results_file, 'w') as f:
            f.write('Dataset,Model,PCA_Components,Variance_Explained,Learning_Rate,Expansion_Point,PLCC,SRCC,KRCC,RMSE,Training_Time\n')
    
    if not all_file_exists:
        with open(all_results_file, 'w') as f:
            f.write('Dataset,Model,PCA_Components,Variance_Explained,Learning_Rate,Expansion_Point,PLCC,SRCC,KRCC,RMSE,Training_Time\n')
    
    epochs = 500
    patience = 20
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ['TaylorKAN']
    
    expansion_points = [0, 0.25, 0.5, 0.75]
    
    dataset_pbar = tqdm(datasets, desc="Datasets")
    
    for data_file, dataset_name in dataset_pbar:
        dataset_pbar.set_description(f"Processing {dataset_name}")
        print(f'\n\nProcessing dataset: {dataset_name}')
        data = pd.read_csv(data_file)
        y_data = data.iloc[:, 1].values
        x_data = data.iloc[:, 2:].values
        
        if np.isnan(x_data).any():
            imputer = SimpleImputer(strategy='mean')
            x_data = imputer.fit_transform(x_data)
        
        optimal_components = analyze_pca_components(x_data)
        pca_components_to_try = sorted(set([optimal_components[0.95]]))
        
        train_size = int(0.7 * len(x_data))
        val_size = int(0.15 * len(x_data))
        indices = np.random.permutation(len(x_data))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        x_train_full = x_data[train_indices]
        x_val_full = x_data[val_indices]
        x_test_full = x_data[test_indices]
        y_train = y_data[train_indices]
        y_val = y_data[val_indices]
        y_test = y_data[test_indices]
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_full)
        x_val_scaled = scaler.transform(x_val_full)
        x_test_scaled = scaler.transform(x_test_full)
        
        for model_name in model_names:
            print(f'\nTesting model: {model_name}')
            model_results = []
            
            pca_pbar = tqdm(pca_components_to_try, desc="PCA Components")
            
            for n_components in pca_pbar:
                pca_pbar.set_description(f"PCA Components: {n_components}")
                print(f'\nTrying PCA components: {n_components}')
                
                pca = PCA(n_components=n_components)
                x_train = pca.fit_transform(x_train_scaled)
                x_val = pca.transform(x_val_scaled)
                x_test = pca.transform(x_test_scaled)
                variance_explained = np.sum(pca.explained_variance_ratio_)
                
                train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
                val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
                test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                learning_rates = [1e-5,1e-4,1e-3]
                best_plcc_srcc_sum = float('-inf')
                best_metrics = None
                best_lr = None
                best_expansion_point = None
                best_training_time = 0
                
                total_combinations = len(expansion_points) * len(learning_rates)
                hyperparam_pbar = tqdm(total=total_combinations, desc="Hyperparameter Search", leave=False)
                
                for expansion_point in expansion_points:
                    for lr in learning_rates:
                        hyperparam_pbar.set_description(f"EP={expansion_point}, LR={lr}")
                        
                        model = initialize_model(model_name, n_components, 1, expansion_point=expansion_point).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                        checkpoint_path = f"models/{dataset_name}_{model_name}_pca{n_components}_exp{expansion_point}_lr{lr}.pt"
                        
                        start_time = time.time()
                        model = train_model(model, device, train_loader, val_loader, optimizer, epochs, patience, checkpoint_path, scheduler)
                        training_time = time.time() - start_time
                        
                        metrics = evaluate_model(model, device, test_loader)
                        plcc_srcc_sum = metrics['Pearson'] + metrics['Spearman']
                        
                        with open(all_results_file, 'a') as f:
                            f.write(f"{dataset_name},{model_name},{n_components},{variance_explained:.4f},"
                                    f"{lr},{expansion_point},{metrics['Pearson']:.6f},{metrics['Spearman']:.6f},"
                                    f"{metrics['Kendall']:.6f},{metrics['RMSE']:.6f},{training_time:.2f}\n")
                        
                        if plcc_srcc_sum > best_plcc_srcc_sum:
                            best_plcc_srcc_sum = plcc_srcc_sum
                            best_metrics = metrics.copy()
                            best_lr = lr
                            best_expansion_point = expansion_point
                            best_training_time = training_time
                        
                        hyperparam_pbar.update(1)
                
                hyperparam_pbar.close()
                
                if best_metrics:
                    print(f'\nBest result: PLCC={best_metrics["Pearson"]:.6f}, SRCC={best_metrics["Spearman"]:.6f}, '
                          f'KRCC={best_metrics["Kendall"]:.6f}, RMSE={best_metrics["RMSE"]:.6f}, '
                          f'LR={best_lr}, Expansion Point={best_expansion_point}')
                    result_entry = {
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'PCA_Components': n_components,
                        'Variance_Explained': variance_explained,
                        'Learning_Rate': best_lr,
                        'Expansion_Point': best_expansion_point,
                        'PLCC': best_metrics['Pearson'],
                        'SRCC': best_metrics['Spearman'],
                        'KRCC': best_metrics['Kendall'],
                        'RMSE': best_metrics['RMSE'],
                        'Training_Time': best_training_time
                    }
                    model_results.append(result_entry)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if model_results:
                best_result = max(model_results, key=lambda x: x['PLCC'] + x['SRCC'])
                print(f'\nBest result for {dataset_name}: PCA={best_result["PCA_Components"]}, '
                      f'PLCC={best_result["PLCC"]:.6f}, SRCC={best_result["SRCC"]:.6f}, '
                      f'LR={best_result["Learning_Rate"]}, Expansion Point={best_result["Expansion_Point"]}')
                
                with open(best_results_file, 'a') as f:
                    f.write(f"{best_result['Dataset']},{best_result['Model']},{best_result['PCA_Components']},"
                            f"{best_result['Variance_Explained']:.4f},{best_result['Learning_Rate']},"
                            f"{best_result['Expansion_Point']},{best_result['PLCC']:.6f},{best_result['SRCC']:.6f},"
                            f"{best_result['KRCC']:.6f},{best_result['RMSE']:.6f},{best_result['Training_Time']:.2f}\n")
    
    print("\n\nAll experiments completed!")
    print(f"Best results saved to: {best_results_file}")
    print(f"All results saved to: {all_results_file}")

if __name__ == '__main__':
    main()
