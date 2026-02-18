import json
import os

from generator import generate_instances
from graph_generator import generate_graph, generate_multiplex_graph


ROOT_DIR = "data"
ADJ_DIR = os.path.join(ROOT_DIR, "custom_graphs")
MULTIPLEX_DIR = os.path.join(ROOT_DIR, "custom_multiplex_graphs")


def main():
    os.makedirs(ADJ_DIR, exist_ok=True)
    os.makedirs(MULTIPLEX_DIR, exist_ok=True)

    instances = generate_instances()

    for idx, instance in enumerate(instances):
        # ----------------------------
        # Adjacency graph
        # ----------------------------
        adj_graph = generate_graph(instance)
        adj_path = os.path.join(ADJ_DIR, f"graph_{idx}.json")

        with open(adj_path, "w", encoding="utf-8") as f:
            json.dump(adj_graph, f, indent=2)

        # ----------------------------
        # Multiplex graph
        # ----------------------------
        multiplex_graph = generate_multiplex_graph(instance)
        multiplex_path = os.path.join(MULTIPLEX_DIR, f"multiplex_{idx}.json")

        with open(multiplex_path, "w", encoding="utf-8") as f:
            json.dump(multiplex_graph, f, indent=2)

        if idx % 500 == 0:
            print(f"Saved instance {idx}")


if __name__ == "__main__":
    main()
