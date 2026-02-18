import json
import os



ROOT_DIR = "data"
ADJ_DIR = os.path.join(ROOT_DIR, "custom_graphs")
MULTIPLEX_DIR = os.path.join(ROOT_DIR, "custom_multiplex_graphs")


def load_graph_pair(index: int):
    """
    Load adjacency and multiplex graphs for a given instance index.

    Returns:
        tuple: (adjacency_graph: dict, multiplex_graph: dict)
    """
    adj_path = os.path.join(ADJ_DIR, f"graph_{index}.json")
    multiplex_path = os.path.join(MULTIPLEX_DIR, f"multiplex_{index}.json")

    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"Adjacency graph not found: {adj_path}")

    if not os.path.exists(multiplex_path):
        raise FileNotFoundError(f"Multiplex graph not found: {multiplex_path}")

    with open(adj_path, "r", encoding="utf-8") as f:
        adjacency_graph = json.load(f)

    with open(multiplex_path, "r", encoding="utf-8") as f:
        multiplex_graph = json.load(f)

    return adjacency_graph, multiplex_graph


def load_all_graph_pairs():
    """
    Load all graph pairs in sorted order.

    Returns:
        list of tuples: [(adj_graph, multiplex_graph), ...]
    """
    indices = []

    for filename in os.listdir(ADJ_DIR):
        if filename.startswith("graph_") and filename.endswith(".json"):
            idx = int(filename.replace("graph_", "").replace(".json", ""))
            indices.append(idx)

    indices.sort()

    graph_pairs = []
    for idx in indices:
        graph_pairs.append(load_graph_pair(idx))

    return graph_pairs


