# src/train.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from src.config import Config
from src.model import GraphSAGE


def prepare_data(G, tfidf_features, labels):

    node_features_list = []
    for node in G.nodes():
        base_features = G.nodes[node]['features']
        tfidf_feature = tfidf_features[node]
        combined = np.concatenate([base_features, tfidf_feature])
        node_features_list.append(combined)

    x = torch.tensor(node_features_list, dtype=torch.float)

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    transform = RandomNodeSplit(num_val=Config.VAL_RATIO, num_test=Config.TEST_RATIO)
    data = transform(data)

    print(f"data is ready nodes: {data.num_nodes}, edges: {data.num_edges}")
    print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")

    return data


def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    weight_tensor = torch.tensor(weights, dtype=torch.float).to(Config.DEVICE)
    return weight_tensor


def train_model(model, data, class_weights):
    model = model.to(Config.DEVICE)
    data = data.to(Config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
        model.train()

        if epoch % Config.PRINT_EVERY == 0:
            print(f'Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, data):
    model.eval()
    data = data.to(Config.DEVICE)
    with torch.no_grad():
        pred = model(data).argmax(dim=1)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"\n===Test result:===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"F1-Weighted: {f1_weighted:.4f}")

    return acc, f1_macro, f1_weighted