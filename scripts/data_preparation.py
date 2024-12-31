import os
import pandas as pd
import networkx as nx
import pickle
from sklearn.preprocessing import StandardScaler

def prepare_data():
    # Definir las rutas relativas para los archivos
    base_path = '/home/neo/Fnetwork/archive (1)/elliptic_bitcoin_dataset'
    classes_path = os.path.join(base_path, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(base_path, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(base_path, 'elliptic_txs_features.csv')

    # Verificar si los archivos existen
    if not (os.path.exists(classes_path) and os.path.exists(edgelist_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Uno o más archivos necesarios no se encuentran en las rutas especificadas.")

    # Cargar los archivos CSV
    try:
        classes_df = pd.read_csv(classes_path)
        edgelist_df = pd.read_csv(edgelist_path)
        features_df = pd.read_csv(features_path, header=None)
    except Exception as e:
        raise ValueError(f"Error al cargar los archivos CSV: {e}")

    # Verificar y mostrar los valores nulos en las clases
    print("Revisando valores nulos en las clases...")
    if classes_df.isnull().values.any():
        print(f"Se encontraron valores nulos en las clases. Número de valores nulos: {classes_df.isnull().sum()}")
        # Mostrar las filas con valores nulos
        print(classes_df[classes_df.isnull().any(axis=1)])

    # Eliminar transacciones con clase 'unknown'
    classes_df = classes_df[classes_df['class'] != 'unknown']
    print(f"Después de eliminar valores 'unknown', quedan {len(classes_df)} transacciones.")

    # Reemplazar las clases '2' por 1 y '0' por 0
    classes_df['class'] = classes_df['class'].map({'2': 1, '0': 0})

    # Depuración: Verificar las columnas de edgelist_df
    print(f"Columnas de edgelist_df: {edgelist_df.columns}")
    print(f"Primeras filas de edgelist_df:\n{edgelist_df.head()}")

    # Verificar si las columnas txId1 y txId2 están presentes
    if 'txId1' not in edgelist_df.columns or 'txId2' not in edgelist_df.columns:
        raise KeyError("'txId1' o 'txId2' no se encuentran en las columnas de edgelist_df.")

    # Verificar si los nodos en el grafo tienen una clase asignada
    missing_nodes = list(set(edgelist_df['txId1'].unique()).union(set(edgelist_df['txId2'].unique())) - set(classes_df['txId'].values))
    print(f"Se encontraron {len(missing_nodes)} nodos en el grafo sin clase asignada.")

    # Asignar clase predeterminada (por ejemplo, '0') a los nodos que no tienen clase
    classes_df = classes_df.set_index('txId')
    classes_df = classes_df.reindex(edgelist_df['txId1'].unique().tolist() + edgelist_df['txId2'].unique().tolist(), fill_value=0).reset_index()

    # Verificar cuántas filas quedan después de eliminar las transacciones con 'unknown'
    print(f"Después de limpiar, quedan {len(classes_df)} transacciones.")

    # Verificar valores nulos en el DataFrame de clases después de la reindexación
    print(f"Después de asignar valores por defecto, aún quedan {classes_df.isnull().sum().sum()} valores nulos en las clases.")

    # Eliminar filas con valores nulos en la columna 'class'
    classes_df.dropna(subset=['class'], inplace=True)

    # Verificar si hay filas restantes después de eliminar nulos
    if classes_df.empty:
        raise ValueError("El DataFrame de clases está vacío después de eliminar las filas nulas.")

    # Normalizar las características
    features_array = features_df.values
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_array)

    # Crear grafo
    G = nx.Graph()
    edges = [(row['txId1'], row['txId2']) for _, row in edgelist_df.iterrows()]
    G.add_edges_from(edges)

    # Verificar que el grafo tenga nodos y aristas
    if G.number_of_nodes() == 0:
        raise ValueError("El grafo no tiene nodos.")
    if G.number_of_edges() == 0:
        raise ValueError("El grafo no tiene aristas.")

    # Asignar características normalizadas a los nodos
    node_list = list(G.nodes())
    for idx, node in enumerate(node_list):
        G.nodes[node]['feature'] = features_normalized[idx]

    # Asignar etiquetas a los nodos
    labels_dict = {row['txId']: row['class'] for _, row in classes_df.iterrows()}
    for node in G.nodes():
        G.nodes[node]['label'] = labels_dict.get(node, 0)  # Usar 0 como clase predeterminada si no tiene etiqueta

    # Verificar si ya existe el grafo procesado
    if os.path.exists('data/graph.pkl'):
        print("El grafo ya existe, cargando...")
    else:
        # Guardar el grafo procesado
        os.makedirs('data', exist_ok=True)  # Crear directorio 'data' si no existe
        with open('data/graph.pkl', 'wb') as f:
            pickle.dump(G, f)
        print(f"Grafo creado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

