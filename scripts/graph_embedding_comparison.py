import os
import torch
from torch_geometric.nn import GATConv, VGAE
from torch_geometric.transforms import RandomLinkSplit
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score

# Crear la carpeta results si no existe
os.makedirs("results", exist_ok=True)

def train_gat(data, hidden_dim=64, num_heads=4, num_epochs=100, lr=0.005):
    """
    Entrena un modelo GAT (Graph Attention Network) para generar embeddings.

    Args:
        data: Objeto Data de PyTorch Geometric.
        hidden_dim: Dimensión oculta para GAT.
        num_heads: Número de cabezas de atención.
        num_epochs: Número de épocas.
        lr: Tasa de aprendizaje.

    Returns:
        Embeddings generados por el modelo GAT.
    """
    class GAT(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GAT, self).__init__()
            self.conv1 = GATConv(in_channels, hidden_dim, heads=num_heads, concat=True, dropout=0.6)
            self.conv2 = GATConv(hidden_dim * num_heads, out_channels, heads=1, concat=False, dropout=0.6)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(data.num_features, 64).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    return embeddings.cpu().detach()

def train_gae(data, hidden_dim=64, num_epochs=100, lr=0.005):
    """
    Entrena un modelo GAE (Graph Autoencoder) para generar embeddings.

    Args:
        data: Objeto Data de PyTorch Geometric.
        hidden_dim: Dimensión oculta para GAE.
        num_epochs: Número de épocas.
        lr: Tasa de aprendizaje.

    Returns:
        Embeddings generados por el modelo GAE.
    """
    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super(Encoder, self).__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=1, concat=True)
            self.conv_mu = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
            self.conv_logstd = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            mu = self.conv_mu(x, edge_index)
            logstd = self.conv_logstd(x, edge_index)
            return mu, logstd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Usar RandomLinkSplit para dividir las aristas
    transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False)
    train_data, _, _ = transform(data)

    encoder = Encoder(data.num_features, hidden_dim).to(device)
    model = VGAE(encoder).to(device)

    train_data = train_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.edge_index)
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)

    return z.cpu().detach()

def visualize_embeddings(embeddings, labels, method_name):
    """
    Visualiza embeddings usando T-SNE.

    Args:
        embeddings: Embeddings generados por un modelo.
        labels: Etiquetas de los nodos.
        method_name: Nombre del método de embeddings (e.g., GAT, GAE).
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
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
    plt.title(f"T-SNE Visualization of {method_name} Embeddings")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    plt.savefig(f"results/tsne_{method_name.lower()}_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualización de {method_name} guardada en 'results/tsne_{method_name.lower()}_visualization.png'.")

if __name__ == "__main__":
    from scripts.data_balancing import balance_data

    print("Cargando datos balanceados...")
    data = balance_data()

    print("Entrenando modelo GAT para embeddings...")
    gat_embeddings = train_gat(data)
    visualize_embeddings(gat_embeddings.numpy(), data.y.cpu().numpy(), "GAT")

    print("Entrenando modelo GAE para embeddings...")
    gae_embeddings = train_gae(data)
    visualize_embeddings(gae_embeddings.numpy(), data.y.cpu().numpy(), "GAE")
