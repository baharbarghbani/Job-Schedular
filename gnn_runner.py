# use evaluate_gnn_policy(instance) function

import random
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
import os


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


def train_gnn_policy(
    training_instances,
    num_train_episodes=500,
    hidden_dim=32,
    learning_rate=1e-3,
    save_path="gnn_policy.pt",
    verbose=True
):
    """
    Train a GNN policy on the given training instances.
    
    Args:
        training_instances: List of JobShopInstance objects for training
        num_train_episodes: Number of training episodes
        hidden_dim: Hidden dimension for GNN layers
        learning_rate: Learning rate for optimizer
        save_path: Path to save the trained model
        verbose: Whether to print training progress
    
    Returns:
        policy: Trained GNN policy
    """
    # Determine input dimension from first instance
    env_test = make_env(training_instances[0])
    obs_test, _ = env_test.reset()
    input_dim = obs_test[ObservationSpaceKey.OPERATIONS.value].shape[1]
    
    # Initialize policy and optimizer
    policy = SimpleGNNPolicy(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    if verbose:
        print(f"Training GNN policy for {num_train_episodes} episodes...")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Training on {len(training_instances)} instances")
        print(f"Model will be saved to: {save_path}")
    
    # Training loop
    for episode in range(num_train_episodes):
        # Sample random training instance
        train_instance = training_instances[episode % len(training_instances)]
        env = make_env(train_instance)
        
        obs, info = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        # Collect trajectory
        while not done:
            action, logp = gnn_action(policy, obs, info)
            obs, reward, done, _, info = env.step(action)
            log_probs.append(logp)
            rewards.append(reward)
        
        # Calculate return
        R = sum(rewards)
        
        # REINFORCE loss
        loss = -R * torch.stack(log_probs).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and episode % 50 == 0:
            makespan = env.current_makespan()
            print(f"Episode {episode}/{num_train_episodes} | Makespan: {makespan} | Return: {R:.2f}")
    
    # Save the trained model
    torch.save({
        'model_state_dict': policy.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
    }, save_path)
    
    if verbose:
        print(f"\nTraining complete! Model saved to {save_path}")
    
    return policy


def evaluate_gnn_policy(
    instance,
    model_path="gnn_policy.pt",
    render_mode=None,
    verbose=True
):
    """
    Evaluate a trained GNN policy on a given instance.
    
    Args:
        instance: JobShopInstance to evaluate on
        model_path: Path to the saved model
        render_mode: Rendering mode (e.g., "save_gif")
        verbose: Whether to print evaluation results
    
    Returns:
        makespan: Final makespan achieved by the policy
    """
    # Load model checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path)
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']
    
    # Initialize policy and load weights
    policy = SimpleGNNPolicy(input_dim=input_dim, hidden_dim=hidden_dim)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()  # Set to evaluation mode
    
    if verbose:
        print(f"Loaded model from {model_path}")
        print(f"Input dimension: {input_dim}, Hidden dimension: {hidden_dim}")
        print("Evaluating on instance...")
    
    # Evaluate on the instance
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
    # Generate instances
    all_instances = generate_instances(seed=320)
    
    # Split into training and test
    training_instances = random.sample(all_instances[:-2], len(all_instances[:-2]))  # Use first 100 for training
    test_instance = all_instances[-2] 
    
    # Phase 1: Training
    print("="*60)
    print("PHASE 1: TRAINING")
    print("="*60)
    trained_policy = train_gnn_policy(
        training_instances=training_instances,
        num_train_episodes=5000,
        hidden_dim=32,
        learning_rate=1e-3,
        save_path="gnn_policy.pt",
        verbose=True
    )
    
    # Phase 2: Evaluation
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION")
    print("="*60)
    final_makespan = evaluate_gnn_policy(
        instance=test_instance,
        model_path="gnn_policy.pt",
        render_mode="save_gif",
        verbose=True
    )
    
    print(f"\nGNN Scheduler achieved makespan: {final_makespan}")