# implement schedule net and compare it with random and spt algorithms

from SPT_runner import run_singleJobShopGraphEnv
from job_shop_lib import JobShopInstance, Operation
from job_shop_lib.dispatching.rules import DispatchingRuleSolver
import random
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

from random_makespan import get_makespan_random

def generate_instances(seed=400):
    random.seed(seed)
    instance_list = []
    metadata_list = []
    
    for job_number in [5, 10, 20, 40, 60]:
        for op_per_job in [5, 10, 15, 20, 25]:
            for m in [5, 10, 20]:
                for i in range(30):
                    jobs = []
                    for j in range(job_number):
                        op = []
                        for k in range(op_per_job):
                            op.append(Operation(
                                machines=random.randint(0, m-1), 
                                duration=random.randint(1, 100)
                            ))
                        jobs.append(op)
                    
                    instance = JobShopInstance(
                        jobs=jobs,
                        name=f"J{job_number}_O{op_per_job}_M{m}_I{i}"
                    )
                    instance_list.append(instance)
                    
                    metadata_list.append({
                        'num_jobs': job_number,
                        'ops_per_job': op_per_job,
                        'num_machines': m,
                        'instance_id': i,
                        'total_operations': job_number * op_per_job
                    })
                    
    return instance_list, metadata_list


def solve_single_instance(args):
    """
    Solve a single instance with all methods.
    This function will be called in parallel.
    """
    idx, instance, metadata, get_makespan_random, run_singleJobShopGraphEnv = args
    
    result_row = {
        **metadata,
        'instance_name': instance.name,
    }
    
    # Method 1: Random baseline
    try:
        start_time = time.perf_counter()
        makespan_random = get_makespan_random(instance)
        time_random = time.perf_counter() - start_time
        
        result_row['makespan_random'] = makespan_random
        result_row['time_random'] = time_random
        result_row['status_random'] = 'success'
    except Exception as e:
        result_row['makespan_random'] = None
        result_row['time_random'] = None
        result_row['status_random'] = f'error'
    
    # Method 2: ScheduleNet with SPT
    try:
        start_time = time.perf_counter()
        makespan_schedulenet = run_singleJobShopGraphEnv(instance)
        time_schedulenet = time.perf_counter() - start_time
        
        result_row['makespan_schedulenet_spt'] = makespan_schedulenet
        result_row['time_schedulenet_spt'] = time_schedulenet
        result_row['status_schedulenet_spt'] = 'success'
    except Exception as e:
        result_row['makespan_schedulenet_spt'] = None
        result_row['time_schedulenet_spt'] = None
        result_row['status_schedulenet_spt'] = f'error'
    
    # Method 3: Most Work Remaining
    try:
        solver_mwr = DispatchingRuleSolver(dispatching_rule="most_work_remaining")
        start_time = time.perf_counter()
        schedule_mwr = solver_mwr.solve(instance)
        time_mwr = time.perf_counter() - start_time
        
        result_row['makespan_mwr'] = schedule_mwr.makespan()
        result_row['time_mwr'] = time_mwr
        result_row['status_mwr'] = 'success'
    except Exception as e:
        result_row['makespan_mwr'] = None
        result_row['time_mwr'] = None
        result_row['status_mwr'] = f'error'
    
    # Method 4: SPT (library version)
    try:
        solver_spt = DispatchingRuleSolver(dispatching_rule="shortest_processing_time")
        start_time = time.perf_counter()
        schedule_spt = solver_spt.solve(instance)
        time_spt = time.perf_counter() - start_time
        
        result_row['makespan_spt'] = schedule_spt.makespan()
        result_row['time_spt'] = time_spt
        result_row['status_spt'] = 'success'
    except Exception as e:
        result_row['makespan_spt'] = None
        result_row['time_spt'] = None
        result_row['status_spt'] = f'error'
    
    return result_row


def solve_all_methods_parallel(instances, metadata, get_makespan_random, 
                               run_singleJobShopGraphEnv, num_workers=None):
    """
    Solve all instances in parallel using multiprocessing.
    
    Args:
        instances: List of instances
        metadata: List of metadata dicts
        get_makespan_random: Your random baseline function
        run_singleJobShopGraphEnv: Your ScheduleNet function
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Solving {len(instances)} instances using {num_workers} parallel workers...")
    
    # Prepare arguments for parallel processing
    args_list = [
        (idx, instance, metadata[idx], get_makespan_random, run_singleJobShopGraphEnv)
        for idx, instance in enumerate(instances)
    ]
    
    # Use multiprocessing Pool
    with Pool(processes=num_workers) as pool:
        results = []
        # Use imap for progress tracking
        for result in pool.imap(solve_single_instance, args_list, chunksize=10):
            results.append(result)
            if len(results) % 500 == 0:
                print(f"Progress: {len(results)}/{len(instances)}")
    
    return pd.DataFrame(results)




if __name__ == "__main__":
    start_total = time.perf_counter()
    
    print("Generating instances...")
    instances, metadata = generate_instances(seed=400)
    print(f"Generated {len(instances)} instances")
    
    # Solve in parallel - uses all CPU cores
    results_df = solve_all_methods_parallel(
        instances, 
        metadata, 
        get_makespan_random, 
        run_singleJobShopGraphEnv,
        num_workers=None  # Auto-detect CPU count
    )
    
    total_time = time.perf_counter() - start_total
    
    # Save results
    results_df.to_csv('all_methods_comparison.csv', index=False)
    print(f"\nCompleted in {total_time:.2f} seconds")
    print(f"Results saved to 'all_methods_comparison.csv'")
    
    # Quick summary
    makespan_cols = [col for col in results_df.columns if col.startswith('makespan_')]
    print("\n--- Average Makespans ---")
    for col in makespan_cols:
        avg = results_df[col].mean()
        print(f"{col}: {avg:.2f}")