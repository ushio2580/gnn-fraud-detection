import os
import pickle
import torch
import numpy as np
from node2vec import Node2Vec

def generate_embeddings():
    # Verificar si el archivo de grafo existe
    if not os.path.exists('data/graph.pkl'):
        raise FileNotFoundError("No se encontró el archivo de grafo 'graph.pkl'. Asegúrate de que los datos estén procesados.")

    # Cargar el grafo desde el archivo
    with open('data/graph.pkl', 'rb') as f:
        G = pickle.load(f)

    # Verificar si los embeddings ya existen
    if os.path.exists('data/embeddings.pt'):
        print("Los embeddings ya han sido generados, cargando desde 'embeddings.pt'.")
        return torch.load('data/embeddings.pt')

    # Configurar Node2Vec
    node2vec = Node2Vec(
        G,
        dimensions=128,       # Número de dimensiones de los embeddings
        walk_length=15,       # Longitud de las caminatas aleatorias
        num_walks=20,         # Número de caminatas aleatorias por nodo
        p=1,                  # Parámetro de control de retorno
        q=2,                  # Parámetro de control de simetría
        workers=10            # Número de hilos para paralelizar el entrenamiento
    )

    # Entrenar Node2Vec
    print("Entrenando el modelo Node2Vec...")
    model_node2vec = node2vec.fit()

    # Convertir embeddings a tensor
    embeddings_array = np.array([model_node2vec.wv[str(node)] for node in G.nodes()])
    embeddings = torch.tensor(embeddings_array, dtype=torch.float)

    # Guardar embeddings generados
    os.makedirs('data', exist_ok=True)  # Asegurar que el directorio 'data' exista
    torch.save(embeddings, 'data/embeddings.pt')

    print(f"Embeddings generados para {embeddings.shape[0]} nodos con {embeddings.shape[1]} dimensiones.")
    return embeddings

# Ejecutar la función si se ejecuta este script directamente
if __name__ == "__main__":
    generate_embeddings()
