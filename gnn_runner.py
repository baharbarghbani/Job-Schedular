# Improved GNN Policy for Job Shop Scheduling - Simplified
# Minimal output version: train, evaluate, save

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from job_shop_lib.reinforcement_learning import ObservationSpaceKey
from generator import generate_general_instances
from job_shop_lib.graphs import JobShopGraph
from job_shop_lib.dispatching.feature_observers import FeatureObserverType
from job_shop_lib.reinforcement_learning import RenderConfig
from job_shop_lib.reinforcement_learning._single_job_shop_graph_env import SingleJobShopGraphEnv
import numpy as np
import os


class ImprovedGNNPolicy(nn.Module):
    """Enhanced GNN with value head and deeper architecture."""
    
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=3):
        super().__init__()
        
        # Multi-layer GCN
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Layer normalization for stability
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Separate policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value function for baseline
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, ready_mask, batch=None):
        # Multi-layer GCN with residual connections
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = F.relu(norm(conv(h, edge_index)))
            if i > 0 and h.shape == h_new.shape:
                h = h + h_new  # Residual connection
            else:
                h = h_new
        
        # Policy logits
        logits = self.policy_head(h).squeeze(-1)
        logits = logits.clone()
        logits[~ready_mask] = -1e9
        
        # Value estimate (global pooling)
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long)
        value = self.value_head(global_mean_pool(h, batch)).squeeze()
        
        return logits, value


def make_env(instance, render_mode=None):
    """Create job shop environment."""
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
    """Convert observation to GNN input format with feature normalization."""
    operations = obs[ObservationSpaceKey.OPERATIONS.value]
    x = torch.tensor(operations, dtype=torch.float32)
    
    # Normalize features (except binary ones)
    for i in [1, 2, 4, 5]:  # Normalize continuous features
        if x[:, i].max() > 0:
            x[:, i] = x[:, i] / (x[:, i].max() + 1e-8)
    
    # Create ready mask
    available_ops = info["available_operations_with_ids"]
    ready_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for op_id, _, _ in available_ops:
        ready_mask[op_id] = True
    
    edge_index = torch.tensor(obs[ObservationSpaceKey.EDGE_INDEX.value], dtype=torch.long)
    
    return x, edge_index, ready_mask, available_ops


def sample_action(policy, obs, info):
    """Sample action from policy."""
    x, edge_index, ready_mask, available_ops = obs_to_gnn_input(obs, info)
    
    logits, value = policy(x, edge_index, ready_mask)
    
    # Sample from categorical distribution
    probs = F.softmax(logits, dim=0)
    dist = torch.distributions.Categorical(probs)
    
    sampled_node = dist.sample()
    log_prob = dist.log_prob(sampled_node)
    entropy = dist.entropy()
    
    # Map back to action
    for op_id, machine_id, job_id in available_ops:
        if op_id == sampled_node.item():
            return (job_id, machine_id), log_prob, entropy, value
    
    # Fallback
    op_id, machine_id, job_id = available_ops[0]
    return (job_id, machine_id), log_prob, entropy, value


def greedy_action(policy, obs, info):
    """Select greedy action for evaluation."""
    x, edge_index, ready_mask, available_ops = obs_to_gnn_input(obs, info)
    
    with torch.no_grad():
        logits, _ = policy(x, edge_index, ready_mask)
        best_node = torch.argmax(logits).item()
    
    for op_id, machine_id, job_id in available_ops:
        if op_id == best_node:
            return (job_id, machine_id)
    
    op_id, machine_id, job_id = available_ops[0]
    return (job_id, machine_id)


def train_gnn_policy(
    training_instances,
    num_episodes=1000,
    hidden_dim=64,
    num_layers=3,
    learning_rate=1e-3,
    entropy_coef=0.01,
    value_coef=0.5,
    gamma=0.99,
    save_path="gnn_policy_improved.pt"
):
    """Train GNN policy with Actor-Critic."""
    # Get input dimension
    env_test = make_env(training_instances[0])
    obs_test, _ = env_test.reset()
    input_dim = obs_test[ObservationSpaceKey.OPERATIONS.value].shape[1]
    
    # Initialize policy
    policy = ImprovedGNNPolicy(
        input_dim=input_dim, 
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Track performance
    all_makespans = []
    baseline_makespan = None
    
    for episode in range(num_episodes):
        # Sample instance
        instance = training_instances[episode % len(training_instances)]
        env = make_env(instance)
        
        obs, info = env.reset()
        log_probs = []
        values = []
        entropies = []
        done = False
        
        # Collect trajectory
        while not done:
            action, logp, ent, val = sample_action(policy, obs, info)
            obs, _, done, _, info = env.step(action)
            log_probs.append(logp)
            entropies.append(ent)
            values.append(val)
        
        # Get final makespan
        makespan = env.current_makespan()
        all_makespans.append(makespan)
        
        # Update baseline (exponential moving average)
        if baseline_makespan is None:
            baseline_makespan = makespan
        else:
            baseline_makespan = 0.99 * baseline_makespan + 0.01 * makespan
        
        # Compute advantage
        advantage = -(makespan - baseline_makespan) / (baseline_makespan + 1e-8)
        
        # Proper shape handling
        value_pred = torch.stack(values).mean()  # Scalar
        value_target = torch.tensor(advantage, dtype=torch.float32)  # Scalar
        
        # Compute losses
        policy_loss = -(torch.stack(log_probs).sum() * advantage)
        value_loss = F.mse_loss(value_pred, value_target)
        entropy_loss = -torch.stack(entropies).mean()
        
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
    
    # Save model
    final_avg = np.mean(all_makespans[-100:])
    best_makespan = min(all_makespans)
    torch.save({
        'model_state_dict': policy.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'final_avg_makespan': float(final_avg),
        'best_makespan': float(best_makespan),
        'all_makespans': [float(m) for m in all_makespans],
    }, save_path)
    
    return policy


def evaluate_gnn_policy(instance, model_path="gnn_policy_improved.pt"):
    """Evaluate trained policy."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint.get('num_layers', 3)
    
    policy = ImprovedGNNPolicy(
        input_dim=input_dim, 
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    env = make_env(instance)
    obs, info = env.reset()
    done = False
    
    while not done:
        action = greedy_action(policy, obs, info)
        obs, _, done, _, info = env.step(action)
    
    return env.current_makespan()


# Main execution
# if __name__ == "__main__":
#     # Generate instances
#     all_instances = generate_general_instances(num_instances=150, seed=320)
#     training_instances = all_instances[:-5]
#     test_instances = all_instances[-5:]
    
#     # Train
#     print("Training GNN policy...")
#     trained_policy = train_gnn_policy(
#         training_instances=training_instances,
#         num_episodes=1000,
#         hidden_dim=64,
#         num_layers=3,
#         learning_rate=1e-3,
#         entropy_coef=0.01,
#         value_coef=0.5,
#         save_path="gnn_policy_improved.pt"
#     )
    
#     # Evaluate on test set
#     test_makespans = []
#     for instance in test_instances:
#         makespan = evaluate_gnn_policy(
#             instance=instance,
#             model_path="gnn_policy_improved.pt"
#         )
#         test_makespans.append(makespan)
    
#     # Print final results
#     print(f"Training complete. Model saved to: gnn_policy_improved.pt")
#     print(f"Test avg makespan: {np.mean(test_makespans):.2f}")
#     print(f"Test best: {min(test_makespans):.2f}")
