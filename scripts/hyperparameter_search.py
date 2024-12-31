import os
import torch
import itertools
from scripts.data_balancing import balance_data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


# Modelo GraphSAGE con hiperparámetros ajustables
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout_rate):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.conv3 = SAGEConv(hidden_dim // 2, out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x


def train_model_with_params(data, learning_rate, hidden_dim, dropout_rate, device):
    """
    Entrena el modelo GraphSAGE con los hiperparámetros especificados.
    """
    model = GraphSAGE(data.num_features, hidden_dim, 2, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience, patience_counter = 10, 0
    best_model_state = None

    for epoch in range(1, 101):  # Ajustar el número de épocas si es necesario
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluar la pérdida en el conjunto de validación
        model.eval()
        with torch.no_grad():
            val_out = model(data.x.to(device), data.edge_index.to(device))
            val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Cargar el mejor modelo encontrado durante el entrenamiento
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def hyperparameter_search(data, param_grid, device):
    """
    Realiza la búsqueda de hiperparámetros.
    """
    best_metrics = {'auc_roc': 0, 'params': None}
    results = []

    # Generar todas las combinaciones de hiperparámetros
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Probando hiperparámetros: {param_dict}")

        # Entrenar el modelo con los hiperparámetros actuales
        model = train_model_with_params(data, **param_dict, device=device)

        # Evaluar el modelo
        model.eval()
        with torch.no_grad():
            output = model(data.x.to(device), data.edge_index.to(device))
            auc_roc = roc_auc_score(data.y.cpu(), output.softmax(dim=1)[:, 1].cpu())
            auc_pr = average_precision_score(data.y.cpu(), output.softmax(dim=1)[:, 1].cpu())

        # Guardar resultados
        results.append({'params': param_dict, 'auc_roc': auc_roc, 'auc_pr': auc_pr})
        print(f"AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

        # Actualizar los mejores hiperparámetros
        if auc_roc > best_metrics['auc_roc']:
            best_metrics['auc_roc'] = auc_roc
            best_metrics['auc_pr'] = auc_pr
            best_metrics['params'] = param_dict

    # Ordenar resultados por AUC-ROC descendente
    results = sorted(results, key=lambda x: x['auc_roc'], reverse=True)

    # Guardar los resultados en un archivo
    output_file = "results/hyperparameter_results.txt"
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Params: {result['params']}, AUC-ROC: {result['auc_roc']:.4f}, AUC-PR: {result['auc_pr']:.4f}\n")

    print(f"Mejores hiperparámetros: {best_metrics['params']}")
    print(f"Resultados guardados en '{output_file}'")

    return best_metrics


if __name__ == "__main__":
    # Obtener los datos balanceados
    data = balance_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'hidden_dim': [64, 128, 256],
        'dropout_rate': [0.2, 0.4, 0.6]
    }

    # Ejecutar la búsqueda de hiperparámetros
    best_metrics = hyperparameter_search(data, param_grid, device)

    print("Hiperparámetros óptimos encontrados:")
    print(best_metrics)
