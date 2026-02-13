from gnn_runner import evaluate_gnn_policy
from SPT_runner import run_singleJobShopGraphEnv
from generator import generate_instances
from random_makespan import get_makespan_random


instances = generate_instances()

k=0
for i in instances:
    time1 = run_singleJobShopGraphEnv(i)
    time2 = get_makespan_random(i)
    time3 = evaluate_gnn_policy(i)

    print(f"{k}:\t random makespan {time2}\t spt makespan {time1}\t gnn makespan {time3}")
    k+=1