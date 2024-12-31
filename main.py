import os
import torch
from scripts.data_preparation import prepare_data
from scripts.generate_embeddings import generate_embeddings
from scripts.data_balancing import balance_data
from scripts.hyperparameter_search import hyperparameter_search, train_model_with_params  # Importación desde el archivo correcto
from scripts.visualize_embeddings import visualize_embeddings
from scripts.evaluate_model import evaluate_model

def main():
    print("Preparando datos...")
    prepare_data()  # Llama a la función para preparar los datos

    print("Generando embeddings...")
    generate_embeddings()  # Llama a la función para generar los embeddings

    print("Balanceando y preparando datos...")
    data = balance_data()  # Llama a la función para balancear los datos y retornar el objeto `Data`

    # Verificación adicional antes de continuar
    if data is None or not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
        raise ValueError("Los datos procesados son inválidos o incompletos.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paso 1: Realizar búsqueda de hiperparámetros
    print("Iniciando búsqueda de hiperparámetros...")
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'hidden_dim': [64, 128, 256],
        'dropout_rate': [0.2, 0.4, 0.6]
    }
    best_metrics = hyperparameter_search(data, param_grid, device)
    best_params = best_metrics['params']
    print(f"Hiperparámetros óptimos encontrados: {best_params}")

    # Paso 2: Reentrenar el modelo con los mejores hiperparámetros
    print("Reentrenando el modelo con los mejores hiperparámetros...")
    model = train_model_with_params(data, **best_params, device=device)

    # Paso 3: Evaluar el modelo
    print("Evaluando el modelo...")
    evaluate_model(data, model)  # Evalúa el modelo entrenado con los datos

    # Paso 4: Visualizar los embeddings
    print("Generando visualización de T-SNE...")
    visualize_embeddings(data)

    print("Proceso completo. Resultados almacenados en la carpeta 'results'.")

if __name__ == "__main__":
    main()
