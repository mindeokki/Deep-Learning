import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE


def do_tsne(X, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(X)


def do_umap(X, n_components=2):
    umap_model = umap.UMAP(n_components=2, random_state=42)
    return umap_model.fit_transform(X)


if __name__ == "__main__":
    # Load example data
    data = load_iris()
    X = data.data
    y = data.target

    X_tsne = do_tsne(X)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    plt.title('t-SNE Visualization of Iris Dataset')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.show()
