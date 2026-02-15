import matplotlib.pyplot as plt

from gnn_runner import evaluate_gnn_policy
from SPT_runner import run_singleJobShopGraphEnv
from generator import generate_instances
from random_makespan import get_makespan_random

instances = generate_instances()

sum_spt = 0
sum_random = 0
sum_gnn = 0

mean_spt = []
mean_random = []
mean_gnn = []

# --- Matplotlib live setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))

line_rand, = ax.plot([], [], label="Random", linestyle="--")
line_spt,  = ax.plot([], [], label="SPT")
line_gnn,  = ax.plot([], [], label="GNN", linewidth=2)

ax.set_xlabel("Instance")
ax.set_ylabel("Mean Makespan")
ax.set_title("Running Mean Makespan")
ax.legend()
ax.grid(True)

# --- Main loop ---
for k, inst in enumerate(instances, start=1):
    t_spt = run_singleJobShopGraphEnv(inst)
    t_rand = get_makespan_random(inst)
    t_gnn = evaluate_gnn_policy(inst)

    sum_spt += t_spt
    sum_random += t_rand
    sum_gnn += t_gnn

    mean_spt.append(sum_spt / k)
    mean_random.append(sum_random / k)
    mean_gnn.append(sum_gnn / k)

    # Update plot data
    x = range(1, k + 1)
    line_rand.set_data(x, mean_random)
    line_spt.set_data(x, mean_spt)
    line_gnn.set_data(x, mean_gnn)

    ax.relim()
    ax.autoscale_view()

    plt.pause(0.01)  # small pause to refresh plot

    # Minimal console log
    print(f"[{k:03d}] Avg → R={mean_random[-1]:.1f} | "
          f"SPT={mean_spt[-1]:.1f} | GNN={mean_gnn[-1]:.1f}")

plt.ioff()
plt.show()
