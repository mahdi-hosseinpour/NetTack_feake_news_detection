# src/graph.py
import networkx as nx
import numpy as np
import pandas as pd
from src.config import Config


def build_graph(df, Features_vect, vocab1):
    """
    Build graph using NetworkX
    - Nodes: Each statement/claim
    - Node features: Combination of non-textual features + TF-IDF
    - Edges: Connect claims that share the same speaker and subject(s)
    """
    # Filter out rows with invalid labels (if any exist)
    df = df[df['label'] != -1].reset_index(drop=True)
    Features_vect = Features_vect.loc[df.index].reset_index(drop=True)

    G = nx.Graph()

    # Add nodes with non-textual base features (excluding subject(s))
    for idx in range(len(df)):
        base_features = df.drop('subject(s)', axis=1, errors='ignore').iloc[idx].values
        G.add_node(idx, features=base_features)

    # Add edges based on identical speaker and subject(s)
    for idx1 in range(len(df)):
        for idx2 in range(idx1 + 1, len(df)):
            row1 = df.iloc[idx1]
            row2 = df.iloc[idx2]
            if row1['subject(s)'] == row2['subject(s)'] and row1['speaker'] == row2['speaker']:
                G.add_edge(idx1, idx2, weight=1)

    # Combine base features with TF-IDF features
    node_features = {}
    for node in G.nodes():
        base_features = df.drop(columns=vocab1, errors='ignore').iloc[node].values.astype(float)
        tfidf_features = Features_vect.iloc[node].values
        combined_features = np.concatenate([base_features, tfidf_features])
        node_features[node] = combined_features

    nx.set_node_attributes(G, node_features, 'features')

    print("Graph constructed successfully!")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    return G