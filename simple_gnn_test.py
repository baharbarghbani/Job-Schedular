import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from job_shop_lib.reinforcement_learning import ObservationSpaceKey
from generator import generate_instances
from job_shop_lib.graphs import JobShopGraph
from job_shop_lib.dispatching.feature_observers import FeatureObserverType
from job_shop_lib.reinforcement_learning import RenderConfig
from job_shop_lib.reinforcement_learning._single_job_shop_graph_env import SingleJobShopGraphEnv
import numpy as np


class SimpleGNNPolicy(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, ready_mask):
        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        logits = self.policy(h).squeeze(-1)
        logits[~ready_mask] = -1e9
        return F.softmax(logits, dim=0)


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


def obs_to_gnn_input(obs, info):
    # The operations matrix contains per-operation features
    operations = obs[ObservationSpaceKey.OPERATIONS.value]
    
    # Convert to torch tensor
    x = torch.tensor(operations, dtype=torch.float32)
    
    # Create ready mask
    available_ops = info["available_operations_with_ids"]
    ready_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for op_id, _, _ in available_ops:
        ready_mask[op_id] = True
    
    # Get edge index (graph structure)
    edge_index = torch.tensor(obs[ObservationSpaceKey.EDGE_INDEX.value], dtype=torch.long)
    
    return x, edge_index, ready_mask, available_ops


def gnn_action(policy, obs, info):
    x, edge_index, ready_mask, available_ops = obs_to_gnn_input(obs, info)
    probs = policy(x, edge_index, ready_mask)
    
    valid_op_ids = [op_id for op_id, _, _ in available_ops]
    valid_probs = probs[valid_op_ids]
    valid_probs = valid_probs / valid_probs.sum()
    
    dist = torch.distributions.Categorical(valid_probs)
    action_idx = dist.sample()
    op_id, machine_id, job_id = available_ops[action_idx.item()]
    
    return (job_id, machine_id), dist.log_prob(action_idx)


def greedy_gnn_action(policy, obs, info):
    x, edge_index, ready_mask, available_ops = obs_to_gnn_input(obs, info)
    
    with torch.no_grad():
        probs = policy(x, edge_index, ready_mask)
    
    valid_op_ids = [op_id for op_id, _, _ in available_ops]
    valid_probs = probs[valid_op_ids]
    best_idx = torch.argmax(valid_probs).item()
    op_id, machine_id, job_id = available_ops[best_idx]
    
    return (job_id, machine_id)


def run_gnn_scheduler(instance, num_train_episodes=500, verbose=True, render_mode=None):
    """
    Train a GNN-based scheduler and evaluate it on the given instance.
    """
    
    # Determine input dimension from the instance
    env_test = make_env(instance)
    obs_test, _ = env_test.reset()
    input_dim = obs_test[ObservationSpaceKey.OPERATIONS.value].shape[1]
    
    # Initialize policy and optimizer
    policy = SimpleGNNPolicy(input_dim=input_dim, hidden_dim=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # Generate training instances
    training_instances = generate_instances(seed=320)
    
    if verbose:
        print(f"Training GNN policy for {num_train_episodes} episodes...")
        print(f"Input dimension: {input_dim}")
        print(f"Training on {len(training_instances)} generated instances")
    
    # Training loop
    for episode in range(num_train_episodes):
        train_instance = training_instances[episode % len(training_instances)]
        env = make_env(train_instance)
        
        obs, info = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            action, logp = gnn_action(policy, obs, info)
            obs, reward, done, _, info = env.step(action)
            log_probs.append(logp)
            rewards.append(reward)
        
        R = sum(rewards)
        loss = -R * torch.stack(log_probs).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and episode % 50 == 0:
            makespan = env.current_makespan()
            print(f"Episode {episode}/{num_train_episodes} | Makespan: {makespan} | Return: {R:.2f}")
    
    if verbose:
        print("\nTraining complete! Evaluating on target instance...")
    
    # Evaluate on the actual input instance
    env = make_env(instance, render_mode=render_mode)
    obs, info = env.reset()
    done = False
    
    while not done:
        action = greedy_gnn_action(policy, obs, info)
        obs, reward, done, _, info = env.step(action)
    
    makespan = env.current_makespan()
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Final Makespan: {makespan}")
        print(f"{'='*50}")
    
    if render_mode:
        env.render()
    
    return makespan


# Example usage
if __name__ == "__main__":
    # Generate a test instance
    test_instance = generate_instances()[0]
    
    # Run GNN-based scheduler
    final_makespan = run_gnn_scheduler(
        instance=test_instance,
        num_train_episodes=800,
        verbose=True,
        render_mode="save_gif"
    )
    
    print(f"\nGNN Scheduler achieved makespan: {final_makespan}")