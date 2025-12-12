# src/graph.py
import networkx as nx
import numpy as np
from src.config import Config

def build_graph(df):

    G = nx.Graph()

    for idx in df.index:
        node_feature = df.drop(columns=['subject(s)'], errors='ignore').iloc[idx].values
        G.add_node(idx, features=node_feature.astype(float))

    for idx1 in df.index:
        for idx2 in df.index:
            if idx1 < idx2:
                row1 = df.iloc[idx1]
                row2 = df.iloc[idx2]
                if row1['speaker'] == row2['speaker'] and row1['subject(s)'] == row2['subject(s)']:
                    G.add_edge(idx1, idx2, weight=1)

    print(f"make graph_node nums: {G.number_of_nodes()}, edge nums: {G.number_of_edges()}")
    return G