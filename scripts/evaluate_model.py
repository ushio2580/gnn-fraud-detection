import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta para guardar resultados
os.makedirs('results', exist_ok=True)

# Modelo GraphSAGE (idéntico al de entrenamiento)
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 256)
        self.conv2 = SAGEConv(256, 128)
        self.conv3 = SAGEConv(128, 64)
        self.conv4 = SAGEConv(64, out_channels)
        self.dropout = torch.nn.Dropout(0.4)

    # Cambiado para aceptar x y edge_index explícitamente
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        return x

def evaluate_model(data, model=None):
    try:
        # Cargar modelo si no se pasa como argumento
        if model is None:
            model = GraphSAGE(data.x.size(1), 2).to(data.x.device)
            model.load_state_dict(torch.load('models/graphsage_best_model.pth'))
            print("Modelo cargado exitosamente.")

        model.eval()

        # Hacer predicciones (cambiado para pasar x y edge_index explícitamente)
        with torch.no_grad():
            out = model(data.x, data.edge_index)  # Cambiado para pasar atributos explícitamente
            prob = F.softmax(out, dim=1)
            pred = torch.argmax(prob, dim=1)

        # Asegurarse de que haya al menos dos clases en las predicciones
        unique_preds = torch.unique(pred)
        if len(unique_preds) == 1:
            print("Advertencia: Las predicciones contienen solo una clase. No se puede calcular AUC-ROC.")
            auc = None
        else:
            # Calcular métricas si hay al menos dos clases
            auc = roc_auc_score(data.y.cpu().numpy(), prob[:, 1].cpu().numpy())
            print(f"AUC-ROC: {auc:.4f}")

        f1 = f1_score(data.y.cpu().numpy(), pred.cpu().numpy())
        print(f"F1-Score: {f1:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(data.y.cpu().numpy(), pred.cpu().numpy())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig('results/confusion_matrix.png')

        # Guardar métricas
        with open('results/metrics.txt', 'w') as f:
            f.write(f"F1-Score: {f1:.4f}\n")
            if auc is not None:
                f.write(f"AUC-ROC: {auc:.4f}\n")

    except Exception as e:
        print(f"Error en la evaluación: {e}")
