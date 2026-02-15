import csv
import time

from generator import generate_instances
from gnn_runner import evaluate_gnn_policy


OUTPUT_CSV = "gnn_results.csv"


def evaluate_gnn_and_log():
    instances = generate_instances()
    rows = []

    for instance_id, inst in enumerate(instances):

        start = time.perf_counter()

        try:
            makespan = evaluate_gnn_policy(inst)
            status = "success"
        except Exception as e:
            makespan = None
            status = f"fail: {e}"

        solve_time = time.perf_counter() - start

        rows.append({
            "makespan": makespan,
            "solve_time": solve_time,
            "status": status,
            "solver": "GNN",
        })


        if instance_id % 500 == 0:
            print(f"makespan={makespan} | time={solve_time:.4f}s")

    return rows


def write_csv(rows, filename):
    fieldnames = [
        "num_jobs",
        "ops_per_job",
        "num_machines",
        "instance_id",
        "total_operations",
        "instance_name",
        "makespan",
        "solve_time",
        "status",
        "solver",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    results = evaluate_gnn_and_log()
    write_csv(results, OUTPUT_CSV)
    print(f"\n✅ GNN evaluation finished. Results saved to {OUTPUT_CSV}")
