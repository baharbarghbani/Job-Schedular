import random
from job_shop_lib import JobShopInstance
import matplotlib.pyplot as plt

def random_shuffler(instance):
    random.seed(400)
    #randomly shuffle operations 
    all_operations_job_ids = []
    for job in instance.jobs:
        num_ops = len(job)
        job_id = instance.jobs.index(job)
        all_operations_job_ids.extend([job_id]*num_ops)
    random.shuffle(all_operations_job_ids)
    return all_operations_job_ids
            

def random_scheduler(instance : JobShopInstance):
    #randomly shuffle operations and assign start times be careful of disjunctive and conjunctive constraints

    all_operations_job_ids = random_shuffler(instance)
    schedule = {}
    machine_available_time = {m:0 for m in range(instance.num_machines)}
    job_last_op_end_time = {j:0 for j in range(instance.num_jobs)}
    for job_id in all_operations_job_ids:
        job = instance.jobs[job_id]
        op_index = len([op for op in schedule.keys() if op[0]==job_id])
        operation = job[op_index]
        machine_id = operation.machine_id
        duration = operation.duration
        earliest_start_time = max(machine_available_time[machine_id], job_last_op_end_time[job_id])
        schedule[(job_id, op_index)] = earliest_start_time
        machine_available_time[machine_id] = earliest_start_time + duration
        job_last_op_end_time[job_id] = earliest_start_time + duration
    return schedule

if __name__ == "__main__":
    # Example usage
    from job_shop_lib.generation import GeneralInstanceGenerator
    generator = GeneralInstanceGenerator(
        duration_range=(1, 10), seed=42, num_jobs=3, num_machines=4
    )
    random_instance = generator.generate()
    schedule = random_scheduler(random_instance)
    for job in random_instance.jobs:
        print(job)
    for op, start_time in schedule.items():
        print(f"Operation {op} starts at time {start_time}")
        
            
    #plot the schedule with matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab20', random_instance.num_jobs)
    for (job_id, op_index), start_time in schedule.items(): 
        operation = random_instance.jobs[job_id][op_index]
        duration = operation.duration
        ax.broken_barh(
            [(start_time, duration)],
            (job_id * 10, 9),
            facecolors=(colors(job_id)),
            edgecolors=('black')
        )
        ax.text(
            start_time + duration / 2,
            job_id * 10 + 4.5,
            f"J{job_id}O{op_index}",
            ha='center',
            va='center',
            color='white',
            fontsize=8
        )
    ax.set_yticks([i * 10 + 4.5 for i in range(random_instance.num_jobs)])
    ax.set_yticklabels([f"Job {i}" for i in range(random_instance.num_jobs)])
    ax.set_xlabel("Time")
    ax.set_title("Randomly Generated Schedule")
    plt.show()
            
