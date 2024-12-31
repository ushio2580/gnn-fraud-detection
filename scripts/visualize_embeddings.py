import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scripts.data_balancing import balance_data

# Crear la carpeta results si no existe
os.makedirs("results", exist_ok=True)

def visualize_embeddings(data):
    """
    Visualiza los embeddings usando T-SNE.
    
    Args:
        data: Objeto `Data` de PyTorch Geometric que contiene embeddings y etiquetas.
    """
    print("visualización con T-SNE...")
    
    # Verificar que los datos tengan embeddings y etiquetas
    if not hasattr(data, 'x') or not hasattr(data, 'y'):
        raise ValueError("El objeto `data` no contiene los atributos requeridos (`x` y `y`).")
    
    embeddings = data.x.cpu().detach().numpy()  # Convertir los embeddings a numpy
    labels = data.y.cpu().numpy()  # Etiquetas
    
    # Normalizar los embeddings antes de aplicar T-SNE
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # Aplicar T-SNE para reducir las dimensiones a 2
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    reduced_embeddings = tsne.fit_transform(normalized_embeddings)
    
    # Crear el scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="k",
        s=15
    )
    plt.colorbar(scatter, label="Classes (0 = Legit, 1 = Fraud)")
    plt.title("T-SNE Visualization of Embeddings")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    
    # Guardar la visualización en la carpeta `results`
    output_path = "results/tsne_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualización de T-SNE guardada en '{output_path}'.")

if __name__ == "__main__":
    # Obtener los datos balanceados
    data = balance_data()
    
    # Llamar a la función de visualización
    visualize_embeddings(data)
