import random

from IPython.display import clear_output
from job_shop_lib import JobShopInstance
from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.graphs import build_disjunctive_graph

from job_shop_lib.reinforcement_learning import (
    # MakespanReward,
    SingleJobShopGraphEnv,
    ObservationSpaceKey,
    IdleTimeReward,
    ObservationDict,
)
from job_shop_lib.dispatching.feature_observers import (
    FeatureObserverType,
    FeatureType,
)
from job_shop_lib.dispatching import DispatcherObserverConfig


def get_makespan_gnn(instance: JobShopInstance, name="", render=False):

    if name != "":
        instance.name = name
    job_shop_graph = build_disjunctive_graph(instance)
    feature_observer_configs = [
        DispatcherObserverConfig(
            FeatureObserverType.IS_READY,
            kwargs={"feature_types": [FeatureType.JOBS]},
        )
    ]

    env = SingleJobShopGraphEnv(
        job_shop_graph=job_shop_graph,
        feature_observer_configs=feature_observer_configs,
        reward_function_config=DispatcherObserverConfig(IdleTimeReward),
        render_mode="save_video",
        render_config={
            "video_config": {"fps": 4}
        }
    )


    def random_action(observation: ObservationDict) -> tuple[int, int]:
        ready_jobs = []
        for job_id, is_ready in enumerate(
            observation[ObservationSpaceKey.JOBS.value].ravel()
        ):
            if is_ready == 1.0:
                ready_jobs.append(job_id)

        job_id = random.choice(ready_jobs)
        machine_id = -1  # We can use -1 if each operation can only be scheduled
        # on one machine.
        return (job_id, machine_id)


    done = False
    obs, _ = env.reset()
    while not done:
        action = random_action(obs)
        obs, reward, done, *_ = env.step(action)
        # if env.render_mode == "human":
        #     env.render()
        #     from matplotlib import pyplot as plt
        #     plt.close("all")  # closes all open figures
        #     clear_output(wait=True)


    #
    if render and env.render_mode == "save_video" or env.render_mode == "save_gif":
        env.render()

    return env.current_makespan()
