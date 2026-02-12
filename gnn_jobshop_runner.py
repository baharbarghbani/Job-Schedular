from generator import generate_instances
from job_shop_lib.graphs import JobShopGraph
from job_shop_lib.dispatching.feature_observers import FeatureObserverType
from job_shop_lib.reinforcement_learning import RenderConfig
from job_shop_lib.reinforcement_learning._single_job_shop_graph_env import SingleJobShopGraphEnv

def make_env(instance, render_mode=None):
    graph = JobShopGraph(instance)

    return SingleJobShopGraphEnv(
        job_shop_graph=graph,
        feature_observer_configs=[
            FeatureObserverType.IS_READY,
            FeatureObserverType.EARLIEST_START_TIME,
            FeatureObserverType.DURATION,
            FeatureObserverType.IS_SCHEDULED,
            FeatureObserverType.POSITION_IN_JOB,
            FeatureObserverType.REMAINING_OPERATIONS,
            FeatureObserverType.IS_COMPLETED,
        ],
        render_mode=render_mode,
        render_config=RenderConfig(),
        use_padding=True,
    )

def run_singleJobShopGraphEnv(instance):
    env = make_env(instance, render_mode="save_gif")

    obs, info = env.reset()
    done = False

    while not done:
        actions = info["available_operations_with_ids"]

        # simple heuristic: shortest processing time
        _, machine_id, job_id = min(
            actions,
            key=lambda x: env.dispatcher.next_operation(x[2]).duration
        )

        obs, reward, done, _, info = env.step((job_id, machine_id))

    # env.render()
    return env.current_makespan()
