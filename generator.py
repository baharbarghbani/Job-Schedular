
from job_shop_lib import JobShopInstance, Operation

import random

def generate_instances():
    random.seed(400)

    instance_list = []
    for job_number in [5,10,20,40,60]:
        for op_per_job in [5,10,15,20,25]:
            for m in [5,10,20]:
                for i in range(30):
                    jobs = []
                    for j in range(job_number):
                        op = []
                        for k in range(op_per_job):
                            op.append(Operation(machines=random.randint(0, m-1), duration=random.randint(1, 100) ))
                        jobs.append(op)
                    instance = JobShopInstance(jobs=jobs)
                    instance_list.append(instance)
                    
    return instance_list
