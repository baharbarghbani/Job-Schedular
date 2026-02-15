from gnn_runner import evaluate_gnn_policy
from SPT_runner import run_singleJobShopGraphEnv
from generator import generate_instances
from random_makespan import get_makespan_random


instances = generate_instances()

sum1 = 0
sum2 = 0
sum3 = 0

k=1
for i in instances:
    time1 = run_singleJobShopGraphEnv(i)
    time2 = get_makespan_random(i)
    time3 = evaluate_gnn_policy(i)

    sum1+=time1
    sum2+=time2
    sum3+=time3

    print(f"{k}:\t random makespan {time2}\t spt makespan {time1}\t gnn makespan {time3}")
    print(f"{k}:\t random makespan {sum2/k}\t spt makespan {sum1/k}\t gnn makespan {sum3/k}")
    k+=1