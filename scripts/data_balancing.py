import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from imblearn.over_sampling import SMOTE

def balance_data():
    # Cargar el grafo
    with open('data/graph.pkl', 'rb') as f:
        G = pickle.load(f)

    # Verificar que el grafo tiene nodos y etiquetas
    if len(G.nodes()) == 0:
        raise ValueError("El grafo está vacío.")
    
    missing_labels = [node for node in G.nodes() if 'label' not in G.nodes[node]]
    if missing_labels:
        print(f"Los siguientes nodos no tienen etiqueta: {missing_labels}")

    # Verificar las etiquetas de los nodos en el grafo
    etiquetas = [G.nodes[node].get('label', 0.0) for node in G.nodes()]
    print(f"Distribución de etiquetas en el grafo: {set(etiquetas)}")

    # Eliminar nodos con etiquetas nulas o desconocidas
    valid_nodes = [node for node in G.nodes() if G.nodes[node].get('label') not in [None, 'unknown']]
    G = G.subgraph(valid_nodes)
    print(f"Después de limpiar, quedan {len(G.nodes())} nodos en el grafo.")

    # Cargar los embeddings
    embeddings = torch.load('data/embeddings.pt')

    # Verificar que los embeddings tienen la forma correcta
    if embeddings.shape[0] != len(G.nodes()):
        raise ValueError(f"Los embeddings tienen un tamaño incorrecto: {embeddings.shape[0]} nodos, pero el grafo tiene {len(G.nodes())} nodos.")

    # Asignación de etiquetas (ajustado según la distribución observada)
    labels_tensor = torch.tensor(
        [1 if G.nodes[node].get('label', 0.0) == 1.0 else 0 for node in G.nodes()],
        dtype=torch.long
    )

    # Verificar que las etiquetas tienen la misma longitud que los embeddings
    if len(labels_tensor) != len(embeddings):
        raise ValueError(f"Las etiquetas no coinciden con los embeddings. Tienen {len(labels_tensor)} y {len(embeddings)} elementos respectivamente.")

    # Imprimir número de nodos en cada clase
    print(f"Etiquetas 'fraud': {labels_tensor.sum().item()}")
    print(f"Etiquetas 'legit': {(labels_tensor == 0).sum().item()}")

    # Balanceo de clases utilizando SMOTE
    fraud_indices = [i for i, label in enumerate(labels_tensor) if label == 1]
    legit_indices = [i for i, label in enumerate(labels_tensor) if label == 0]

    # Imprimir número de nodos en cada clase antes del balanceo
    print(f"Total de nodos de clase 'fraud': {len(fraud_indices)}")
    print(f"Total de nodos de clase 'legit': {len(legit_indices)}")

    if len(fraud_indices) == 0 or len(legit_indices) == 0:
        raise ValueError("No hay nodos suficientes de una clase para balancear.")

    # Aplicar SMOTE: vamos a usar solo los embeddings (características) para la creación de ejemplos sintéticos
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    embeddings_np = embeddings.cpu().numpy()  # Convertir a numpy para usar con SMOTE
    labels_np = labels_tensor.cpu().numpy()

    # Aplicar SMOTE en las características (embeddings) de los nodos
    embeddings_resampled, labels_resampled = smote.fit_resample(embeddings_np, labels_np)

    # Convertir de vuelta a tensores de PyTorch
    embeddings_resampled = torch.tensor(embeddings_resampled, dtype=torch.float)
    labels_resampled = torch.tensor(labels_resampled, dtype=torch.long)

    # Verificar el balanceo después de aplicar SMOTE
    print(f"Balanceo después de SMOTE:")
    print(f"Etiquetas 'fraud': {labels_resampled.sum().item()}")
    print(f"Etiquetas 'legit': {(labels_resampled == 0).sum().item()}")

    # Crear divisiones para asegurar que ambas clases estén presentes en los dos conjuntos
    fraud_indices_resampled = [i for i, label in enumerate(labels_resampled) if label == 1]
    legit_indices_resampled = [i for i, label in enumerate(labels_resampled) if label == 0]

    # Dividir los datos para asegurar que ambos conjuntos tienen ambas clases
    train_fraud_indices = fraud_indices_resampled[:int(0.8 * len(fraud_indices_resampled))]
    val_fraud_indices = fraud_indices_resampled[int(0.8 * len(fraud_indices_resampled)):]

    train_legit_indices = legit_indices_resampled[:int(0.8 * len(legit_indices_resampled))]
    val_legit_indices = legit_indices_resampled[int(0.8 * len(legit_indices_resampled)):]

    # Unir las divisiones para entrenamiento y validación
    train_indices = train_fraud_indices + train_legit_indices
    val_indices = val_fraud_indices + val_legit_indices

    # Crear máscaras para PyTorch Geometric
    train_mask = torch.zeros(len(labels_resampled), dtype=torch.bool)
    val_mask = torch.zeros(len(labels_resampled), dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True

    # Verificación adicional del balance de clases después del balanceo
    train_labels = labels_resampled[train_indices]
    val_labels = labels_resampled[val_indices]

    # Verificar que ambas clases están presentes en los datos de entrenamiento y validación
    if len(train_labels.unique()) != 2:
        raise ValueError("El conjunto de entrenamiento tiene solo una clase, no se puede entrenar un modelo con solo una clase.")
    
    if len(val_labels.unique()) != 2:
        raise ValueError("El conjunto de validación tiene solo una clase, no se puede evaluar un modelo con solo una clase.")

    # Crear edge_index para PyTorch Geometric
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    edge_index = torch.tensor(
        [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Verificar que el edge_index tiene la forma correcta
    if edge_index.shape[0] != 2:
        raise ValueError(f"El edge_index tiene una forma incorrecta: {edge_index.shape}")

    # Preparar datos
    data = Data(
        x=embeddings_resampled,
        edge_index=edge_index,
        y=labels_resampled,
        train_mask=train_mask,
        val_mask=val_mask
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Verificar que los datos preparados tengan las formas correctas
    if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
        raise ValueError("El objeto 'Data' tiene atributos inválidos o incompletos.")

    # Guardar los datos procesados
    torch.save(data, 'data/processed_data.pt')
    print(f"Datos balanceados y procesados guardados en 'data/processed_data.pt'.")

    # Retornar el objeto `data` para ser utilizado en `main.py`
    return data

# Ejecutar la función si se ejecuta este script directamente
if __name__ == "__main__":
    balance_data()
