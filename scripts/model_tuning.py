import os
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import SAGEConv
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model

# Crear la carpeta results si no existe
os.makedirs("results", exist_ok=True)

def evaluate_with_metrics(model, data, device):
    # Establecemos el modelo en modo de evaluación
    model.eval()
    
    # Realizar predicciones: pasamos el objeto data completo
    output = model(data.to(device))  # Enviar todo el objeto 'data' al modelo
    
    # Obtener las predicciones
    pred = output.argmax(dim=1)

    # Calcular las métricas
    print("Evaluando el modelo...")

    # Obtener las etiquetas verdaderas
    true_labels = data.y.to(device)

    # Reporte de clasificación (Precision, Recall, F1, etc.)
    print("Reporte de clasificación:")
    print(classification_report(true_labels.cpu(), pred.cpu(), target_names=["Legit", "Fraud"]))

    # Matriz de confusión
    print("Matriz de Confusión:")
    cm = confusion_matrix(true_labels.cpu(), pred.cpu())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("results/confusion_matrix.png")  # Guardar el gráfico
    plt.close()  # Cerrar el gráfico para liberar memoria

    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(true_labels.cpu(), output[:, 1].cpu().detach().numpy())
    pr_auc = auc(recall, precision)
    print(f"AUC-PR: {pr_auc:.4f}")
    plt.figure()
    plt.plot(recall, precision, label=f'AUC-PR = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig("results/precision_recall_curve.png")  # Guardar el gráfico
    plt.close()

    # AUC-ROC
    auc_roc = roc_auc_score(true_labels.cpu(), output[:, 1].cpu().detach().numpy())
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Curva ROC
    fpr, tpr, _ = precision_recall_curve(true_labels.cpu(), output[:, 1].cpu().detach().numpy())
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("results/roc_curve.png")  # Guardar el gráfico
    plt.close()

    return auc_roc, pr_auc

def tune_model(data):
    # Verificamos el dispositivo (GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Entrenamos el modelo con los datos balanceados
    model = train_model(data)  # Este método debe entrenar y devolver el modelo entrenado
    
    # Evaluamos el modelo y obtenemos las métricas
    auc_roc, pr_auc = evaluate_with_metrics(model, data, device)

    return model, auc_roc, pr_auc

if __name__ == "__main__":
    # Ejecutar el ajuste del modelo y evaluación
    from scripts.data_balancing import balance_data
    data = balance_data()  # Obtener los datos balanceados
    
    # Ajustar y evaluar el modelo
    model, auc_roc, pr_auc = tune_model(data)
    
    # Imprimir las métricas
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
