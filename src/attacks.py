# src/attacks.py
import torch
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import accuracy_score
from src.train import evaluate_model
from src.config import Config


class Nettack:
    """
    Nettack attack on graph nodes (feature + structure perturbation)
    Simplified version compatible with PyTorch Geometric
    """

    def __init__(self, data, model, target_node, n_perturbations=5):
        self.data = data.to('cpu')
        self.model = model.to('cpu')
        self.model.eval()
        self.target_node = target_node
        self.n_perturbations = n_perturbations
        self.n_nodes = data.num_nodes
        self.n_features = data.x.shape[1]
        self.adj = self._get_adjacency_matrix()
        self.degree = np.array(self.adj.sum(axis=1)).flatten()

    def _get_adjacency_matrix(self):
        row = self.data.edge_index[0].numpy()
        col = self.data.edge_index[1].numpy()
        data = np.ones(len(row))
        adj = coo_matrix((data, (row, col)), shape=(self.n_nodes, self.n_nodes))
        return adj + adj.T  # undirected

    def attack(self):
        """
        Perform Nettack on target node
        Returns perturbed data
        """
        model = self.model
        target = self.target_node
        best_loss = -np.inf
        best_perturbed_adj = self.adj.copy()
        best_perturbed_features = self.data.x.clone()

        candidates = np.arange(self.n_nodes)
        candidates = candidates[candidates != target]

        for _ in range(self.n_perturbations):
            losses = []
            for candidate in candidates:
                # Try adding/removing edge
                if best_perturbed_adj[target, candidate] == 1:
                    # Remove edge
                    perturbed_adj = best_perturbed_adj.copy()
                    perturbed_adj[target, candidate] = 0
                    perturbed_adj[candidate, target] = 0
                else:
                    # Add edge
                    perturbed_adj = best_perturbed_adj.copy()
                    perturbed_adj[target, candidate] = 1
                    perturbed_adj[candidate, target] = 1

                # Convert to edge_index
                row, col = perturbed_adj.nonzero()
                perturbed_edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)

                perturbed_data = self.data.clone()
                perturbed_data.edge_index = perturbed_edge_index

                with torch.no_grad():
                    out = model(perturbed_data)
                    loss = -out[target, self.data.y[target].item()].item()  # maximize wrong class probability

                losses.append((loss, candidate, perturbed_adj))

            # Choose best perturbation
            losses.sort(reverse=True)
            best_loss, best_candidate, best_adj = losses[0]
            best_perturbed_adj = best_adj

        # Final perturbed data
        row, col = best_perturbed_adj.nonzero()
        perturbed_edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
        perturbed_data = self.data.clone()
        perturbed_data.edge_index = perturbed_edge_index

        print(f"Nettack completed on node {self.target_node} with {self.n_perturbations} perturbations")
        return perturbed_data


def nettack_attack(data, model, target_nodes=None, n_perturbations=5):
    """
    Apply Nettack on multiple target nodes
    """
    if target_nodes is None:
        # Randomly select some test nodes
        target_nodes = data.test_mask.nonzero(as_tuple=True)[0][:10]  # attack on 10 test nodes

    perturbed_datas = []
    for target in target_nodes:
        attack = Nettack(data, model, target.item(), n_perturbations=n_perturbations)
        perturbed_data = attack.attack()
        perturbed_datas.append(perturbed_data)

    # For simplicity, return the last perturbed graph (or average, but usually one graph with multiple perturbations)
    print("Nettack attack applied on multiple nodes")
    return perturbed_data


def adversarial_training(model, clean_data, num_epochs=100):
    """
    Adversarial Training with Nettack-generated examples
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Generate adversarial example on the fly
        target_node = clean_data.test_mask.nonzero(as_tuple=True)[0][0].item()  # one target
        attack = Nettack(clean_data, model, target_node, n_perturbations=3)
        adv_data = attack.attack()
        adv_data = adv_data.to(Config.DEVICE)

        out = model(adv_data)
        loss = F.cross_entropy(out[adv_data.train_mask], adv_data.y[adv_data.train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Adversarial Training Epoch {epoch} | Loss: {loss.item():.4f}")

    print("Adversarial Training completed")
    return model


def evaluate_with_nettack(model, data):
    """
    Evaluate model under Nettack attack
    """
    print("\n=== Evaluation under Nettack Attack ===")
    adv_data = nettack_attack(data, model, n_perturbations=5)
    acc, f1_macro, f1_weighted = evaluate_model(model, adv_data)
    return acc, f1_macro, f1_weighted