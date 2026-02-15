"""
ScheduleNet: GNN + DRL for Job Shop Scheduling
Based on the paper "Learn to solve multi-agent scheduling problems with reinforcement learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt

from job_shop_lib.graphs import build_resource_task_graph
from job_shop_lib.visualization.gantt import plot_gantt_chart
from job_shop_lib.reinforcement_learning import (
    SingleJobShopGraphEnv,
    ResourceTaskGraphObservation,
)
from job_shop_lib.dispatching.feature_observers import FeatureObserverType


# ============================================================================
# Type-Aware Graph Attention (TGA) Implementation
# ============================================================================

class MultiplicativeInteraction(nn.Module):
    """Multiplicative Interaction Layer for type-aware embeddings"""
    
    def __init__(self, input_dim: int, context_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        
        # Generate transformation matrices based on context
        self.weight_generator = nn.Linear(context_dim, input_dim * output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, input_dim]
            context: Context embedding [batch_size, context_dim]
        Returns:
            Transformed features [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Generate weights from context
        weights = self.weight_generator(context)  # [batch_size, input_dim * output_dim]
        weights = weights.view(batch_size, self.input_dim, self.output_dim)
        
        # Apply transformation
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        output = torch.bmm(x, weights).squeeze(1)  # [batch_size, output_dim]
        
        return output + self.bias


class TypeAwareGraphAttention(MessagePassing):
    """
    Type-Aware Graph Attention Layer implementing ScheduleNet's TGA
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 32,
        num_node_types: int = 5,
        num_edge_types: int = 3,
    ):
        super().__init__(aggr=None, flow="source_to_target")
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        
        # Type encoders
        self.node_type_encoder = nn.Embedding(num_node_types, hidden_dim)
        self.edge_type_encoder = nn.Embedding(num_edge_types, hidden_dim)
        
        # Edge update with MI layer
        self.edge_mi = MultiplicativeInteraction(
            input_dim=2 * node_dim + edge_dim,
            context_dim=hidden_dim,
            output_dim=hidden_dim
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention with MI layer
        self.attn_mi = MultiplicativeInteraction(
            input_dim=2 * node_dim + edge_dim,
            context_dim=hidden_dim,
            output_dim=hidden_dim
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Node update with MI layer
        self.node_mi = MultiplicativeInteraction(
            input_dim=node_dim + hidden_dim * num_node_types,
            context_dim=hidden_dim,
            output_dim=hidden_dim
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_types: torch.Tensor,
        edge_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            node_types: Node type indices [num_nodes]
            edge_types: Edge type indices [num_edges] (optional)
        Returns:
            Updated node features, updated edge features
        """
        # Type-aware edge update and attention
        edge_attr_new, attn_logits = self.edge_updater(
            edge_index, x=x, edge_attr=edge_attr, 
            node_types=node_types, edge_types=edge_types
        )
        
        # Type-aware message aggregation and node update
        x_new = self.propagate(
            edge_index, x=x, edge_attr=edge_attr_new,
            attn_logits=attn_logits, node_types=node_types
        )
        
        return x_new, edge_attr_new
    
    def edge_update(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        node_types_j: torch.Tensor,
        edge_types: Optional[torch.Tensor] = None,
        index: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update edges with type-awareness"""
        # Get context from source node type
        context = self.node_type_encoder(node_types_j)
        
        # Concatenate features
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Type-aware edge encoding
        edge_encoding = self.edge_mi(edge_input, context)
        edge_new = self.edge_mlp(edge_encoding)
        
        # Type-aware attention logits
        attn_encoding = self.attn_mi(edge_input, context)
        attn_logits = self.attn_mlp(attn_encoding).squeeze(-1)
        
        return edge_new, attn_logits
    
    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        attn_logits: torch.Tensor,
        node_types_j: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute messages with type-aware attention"""
        # Compute attention scores per node type
        attn_weights = self._compute_type_aware_attention(
            attn_logits, node_types_j, index
        )
        
        # Weight edge features by attention
        return attn_weights.unsqueeze(-1) * edge_attr
    
    def _compute_type_aware_attention(
        self,
        attn_logits: torch.Tensor,
        node_types: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention scores normalized per node type"""
        num_nodes = index.max().item() + 1
        
        # Softmax per (target_node, source_type) group
        attn_weights = torch.zeros_like(attn_logits)
        
        for node_idx in range(num_nodes):
            mask = (index == node_idx)
            if not mask.any():
                continue
                
            node_logits = attn_logits[mask]
            node_types_local = node_types[mask]
            
            # Normalize per type
            for type_idx in range(self.num_node_types):
                type_mask = (node_types_local == type_idx)
                if not type_mask.any():
                    continue
                
                type_logits = node_logits[type_mask]
                type_weights = F.softmax(type_logits, dim=0)
                
                # Map back to global indices
                global_mask = mask.clone()
                local_indices = torch.where(mask)[0]
                type_local_indices = local_indices[type_mask]
                
                attn_weights[type_local_indices] = type_weights
        
        return attn_weights
    
    def aggregate(
        self,
        messages: torch.Tensor,
        node_types: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate messages per node type"""
        num_nodes = index.max().item() + 1
        
        # Aggregate per type
        aggregated = torch.zeros(
            num_nodes, 
            self.num_node_types * self.hidden_dim,
            device=messages.device
        )
        
        for type_idx in range(self.num_node_types):
            type_mask = (node_types == type_idx)
            
            # Sum messages of this type
            type_messages = torch.zeros(
                num_nodes, self.hidden_dim, device=messages.device
            )
            type_messages.index_add_(0, index[type_mask], messages[type_mask])
            
            # Place in aggregated tensor
            start_idx = type_idx * self.hidden_dim
            end_idx = (type_idx + 1) * self.hidden_dim
            aggregated[:, start_idx:end_idx] = type_messages
        
        return aggregated
    
    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        """Update nodes with aggregated messages"""
        # Get context from node type
        context = self.node_type_encoder(node_types)
        
        # Concatenate node features and aggregated messages
        node_input = torch.cat([x, aggr_out], dim=-1)
        
        # Type-aware node update
        node_encoding = self.node_mi(node_input, context)
        x_new = self.node_mlp(node_encoding)
        
        return x_new


# ============================================================================
# ScheduleNet Model
# ============================================================================

class ScheduleNet(nn.Module):
    """
    ScheduleNet: GNN-based scheduler for job shop scheduling
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_node_types: int = 5,
        num_edge_types: int = 3,
    ):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # TGA layers
        self.tga_layers = nn.ModuleList([
            TypeAwareGraphAttention(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
            )
            for _ in range(num_layers)
        ])
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(3 * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_types: torch.Tensor,
        agent_idx: int,
        feasible_tasks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_feature_dim]
            node_types: [num_nodes] - type index for each node
            agent_idx: Index of the idle agent
            feasible_tasks: [num_feasible_tasks] - indices of feasible tasks
        Returns:
            action_logits: [num_feasible_tasks] - logits for each feasible action
        """
        # Encode features
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        
        # Apply TGA layers
        for tga_layer in self.tga_layers:
            x, edge_attr = tga_layer(x, edge_index, edge_attr, node_types)
        
        # Compute assignment logits
        agent_embedding = x[agent_idx].unsqueeze(0)  # [1, hidden_dim]
        task_embeddings = x[feasible_tasks]  # [num_feasible_tasks, hidden_dim]
        
        # Get edge embeddings from agent to tasks
        # Find edges from agent to each feasible task
        edge_embeddings = self._get_edge_embeddings(
            edge_index, edge_attr, agent_idx, feasible_tasks
        )
        
        # Concatenate agent, task, and edge embeddings
        num_feasible = feasible_tasks.size(0)
        agent_repeated = agent_embedding.repeat(num_feasible, 1)
        
        combined = torch.cat([
            agent_repeated, 
            task_embeddings, 
            edge_embeddings
        ], dim=-1)
        
        # Compute logits
        logits = self.actor(combined).squeeze(-1)
        
        return logits
    
    def _get_edge_embeddings(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        agent_idx: int,
        task_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Extract edge embeddings from agent to tasks"""
        # Find edges where source is agent_idx
        source_mask = (edge_index[0] == agent_idx)
        
        edge_embeddings = []
        for task_idx in task_indices:
            # Find edge from agent to this task
            edge_mask = source_mask & (edge_index[1] == task_idx)
            
            if edge_mask.any():
                edge_emb = edge_attr[edge_mask][0]
            else:
                # If no edge exists, use zeros
                edge_emb = torch.zeros(edge_attr.size(1), device=edge_attr.device)
            
            edge_embeddings.append(edge_emb)
        
        return torch.stack(edge_embeddings)


# ============================================================================
# Environment Wrapper for ScheduleNet
# ============================================================================

class ScheduleNetEnv:
    """
    Environment wrapper that converts job_shop_lib format to ScheduleNet format
    """
    
    def __init__(self, instance):
        self.instance = instance
        self.env = SingleJobShopGraphEnv(instance)
        self.reset()
        
    def reset(self):
        """Reset environment"""
        obs, info = self.env.reset()
        self.current_obs = obs
        return self._convert_observation(obs)
    
    def step(self, action: int):
        """
        Take a step in the environment
        
        Args:
            action: Index of the selected task (relative to feasible tasks)
        Returns:
            observation, reward, done, info
        """
        # Get actual action from feasible tasks
        feasible_tasks = self.current_obs.get_feasible_operations()
        actual_action = feasible_tasks[action]
        
        obs, reward, done, truncated, info = self.env.step(actual_action)
        self.current_obs = obs
        
        return self._convert_observation(obs), reward, done or truncated, info
    
    def _convert_observation(self, obs):
        """Convert observation to ScheduleNet format"""
        graph_data = self._build_graph(obs)
        
        return {
            'graph': graph_data,
            'agent_idx': self._get_idle_agent(obs),
            'feasible_tasks': self._get_feasible_task_indices(obs),
            'makespan': obs.current_time,
        }
    
    def _build_graph(self, obs) -> Data:
        """Build PyTorch Geometric graph from observation"""
        # Build resource-task graph
        rt_graph = build_resource_task_graph(self.instance)
        
        # Extract node features
        node_features = self._extract_node_features(obs)
        
        # Extract edge features
        edge_index, edge_features = self._extract_edge_features(obs, rt_graph)
        
        # Get node types
        node_types = self._get_node_types(obs)
        
        return Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_features),
            node_types=torch.LongTensor(node_types),
        )
    
    def _extract_node_features(self, obs) -> np.ndarray:
        """Extract node features for agents and tasks"""
        num_machines = len(self.instance.machines)
        num_operations = sum(len(job.operations) for job in self.instance.jobs)
        
        features = []
        
        # Agent (machine) features
        for machine_id in range(num_machines):
            is_idle = self._is_machine_idle(obs, machine_id)
            current_time = obs.current_time
            
            features.append([
                1.0,  # is_agent
                1.0 if is_idle else 0.0,  # is_idle
                0.0,  # is_assigned
                current_time,
                0.0,  # processing_time (not applicable for agents)
            ])
        
        # Task (operation) features
        for job in self.instance.jobs:
            for op_idx, operation in enumerate(job.operations):
                is_processable = operation.operation_id in obs.get_feasible_operations()
                is_completed = self._is_operation_completed(obs, operation.operation_id)
                
                wait_time = self._get_operation_wait_time(obs, operation.operation_id)
                processing_time = operation.duration
                remaining_ops = len(job.operations) - op_idx - 1
                completion_ratio = op_idx / len(job.operations)
                
                features.append([
                    0.0,  # is_agent
                    0.0,  # is_idle (not applicable for tasks)
                    1.0 if is_completed else 0.0,  # is_assigned/completed
                    wait_time,
                    processing_time,
                ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_edge_features(self, obs, rt_graph) -> Tuple[np.ndarray, np.ndarray]:
        """Extract edge indices and features"""
        edge_list = []
        edge_features = []
        
        num_machines = len(self.instance.machines)
        
        # Machine to operation edges
        for machine_id in range(num_machines):
            for job in self.instance.jobs:
                for operation in job.operations:
                    if operation.machine_id == machine_id:
                        task_idx = num_machines + operation.operation_id
                        
                        # Edge from machine to operation
                        edge_list.append([machine_id, task_idx])
                        is_processable = operation.operation_id in obs.get_feasible_operations()
                        edge_features.append([1.0 if is_processable else 0.0])
                        
                        # Edge from operation to machine (bidirectional)
                        edge_list.append([task_idx, machine_id])
                        edge_features.append([1.0 if is_processable else 0.0])
        
        # Operation precedence edges
        for job in self.instance.jobs:
            for i in range(len(job.operations) - 1):
                op1_idx = num_machines + job.operations[i].operation_id
                op2_idx = num_machines + job.operations[i + 1].operation_id
                
                edge_list.append([op1_idx, op2_idx])
                edge_features.append([1.0])  # Precedence edge
                
                edge_list.append([op2_idx, op1_idx])
                edge_features.append([0.0])  # Reverse precedence
        
        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_features = np.array(edge_features, dtype=np.float32)
        
        return edge_index, edge_features
    
    def _get_node_types(self, obs) -> np.ndarray:
        """Get node type for each node"""
        num_machines = len(self.instance.machines)
        num_operations = sum(len(job.operations) for job in self.instance.jobs)
        
        node_types = []
        
        # Machine types
        for machine_id in range(num_machines):
            is_idle = self._is_machine_idle(obs, machine_id)
            if is_idle:
                node_types.append(1)  # unassigned-agent
            else:
                node_types.append(0)  # assigned-agent
        
        # Operation types
        for job in self.instance.jobs:
            for operation in job.operations:
                is_completed = self._is_operation_completed(obs, operation.operation_id)
                is_processable = operation.operation_id in obs.get_feasible_operations()
                
                if is_completed:
                    node_types.append(2)  # assigned-task (completed)
                elif is_processable:
                    node_types.append(3)  # processable-task
                else:
                    node_types.append(4)  # unprocessable-task
        
        return np.array(node_types, dtype=np.int64)
    
    def _is_machine_idle(self, obs, machine_id: int) -> bool:
        """Check if machine is idle"""
        # Check if any operation is currently being processed on this machine
        for job in self.instance.jobs:
            for operation in job.operations:
                if operation.machine_id == machine_id:
                    if hasattr(obs, 'running_operations'):
                        if operation.operation_id in obs.running_operations:
                            return False
        return True
    
    def _is_operation_completed(self, obs, operation_id: int) -> bool:
        """Check if operation is completed"""
        if hasattr(obs, 'completed_operations'):
            return operation_id in obs.completed_operations
        return False
    
    def _get_operation_wait_time(self, obs, operation_id: int) -> float:
        """Get wait time for an operation"""
        # Simplified: return 0 for now
        return 0.0
    
    def _get_idle_agent(self, obs) -> int:
        """Get index of idle agent (machine)"""
        for machine_id in range(len(self.instance.machines)):
            if self._is_machine_idle(obs, machine_id):
                return machine_id
        return 0  # Default to first machine
    
    def _get_feasible_task_indices(self, obs) -> np.ndarray:
        """Get indices of feasible tasks in the graph"""
        num_machines = len(self.instance.machines)
        feasible_ops = obs.get_feasible_operations()
        
        # Convert operation IDs to node indices
        task_indices = [num_machines + op_id for op_id in feasible_ops]
        
        return np.array(task_indices, dtype=np.int64)


# ============================================================================
# Clip-REINFORCE Training Algorithm
# ============================================================================

class ClipREINFORCE:
    """
    Clip-REINFORCE training algorithm (PPO without value function)
    """
    
    def __init__(
        self,
        model: ScheduleNet,
        learning_rate: float = 3e-4,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        polyak: float = 0.1,
    ):
        self.model = model
        self.baseline_model = type(model)(
            **{k: v for k, v in model.__dict__.items() if not k.startswith('_')}
        )
        self.baseline_model.load_state_dict(model.state_dict())
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.polyak = polyak
        
        self.training_stats = defaultdict(list)
        
    def compute_normalized_return(
        self,
        makespan_policy: float,
        makespan_baseline: float,
        tau: int,
        T: int,
    ) -> float:
        """Compute normalized return (Equation 8 in paper)"""
        normalized_makespan = (makespan_policy - makespan_baseline) / makespan_baseline
        normalized_return = -self.gamma ** (T - tau) * normalized_makespan
        
        return normalized_return
    
    def train_episode(
        self,
        env: ScheduleNetEnv,
        num_inner_updates: int = 5,
    ) -> Dict:
        """Train on a single episode"""
        # Collect trajectory with current policy
        trajectory = self._collect_trajectory(env, self.model)
        
        # Collect baseline trajectory
        baseline_trajectory = self._collect_trajectory(env, self.baseline_model)
        
        makespan_policy = trajectory['makespan']
        makespan_baseline = baseline_trajectory['makespan']
        
        # Compute returns for each step
        T = len(trajectory['states'])
        returns = []
        
        for tau in range(T):
            ret = self.compute_normalized_return(
                makespan_policy, makespan_baseline, tau, T
            )
            returns.append(ret)
        
        returns = torch.FloatTensor(returns)
        
        # Perform multiple updates
        total_loss = 0.0
        
        for _ in range(num_inner_updates):
            loss = self._update_step(trajectory, returns)
            total_loss += loss
        
        # Polyak averaging for baseline
        self._update_baseline()
        
        # Track statistics
        stats = {
            'makespan': makespan_policy,
            'baseline_makespan': makespan_baseline,
            'loss': total_loss / num_inner_updates,
            'num_steps': T,
        }
        
        return stats
    
    def _collect_trajectory(self, env: ScheduleNetEnv, model: ScheduleNet) -> Dict:
        """Collect a trajectory using the given model"""
        states = []
        actions = []
        log_probs = []
        
        obs = env.reset()
        done = False
        
        while not done:
            # Get action from model
            with torch.no_grad():
                logits = model(
                    obs['graph'].x,
                    obs['graph'].edge_index,
                    obs['graph'].edge_attr,
                    obs['graph'].node_types,
                    obs['agent_idx'],
                    torch.LongTensor(obs['feasible_tasks']),
                )
                
                probs = F.softmax(logits, dim=0)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            # Store transition
            states.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            
            # Take step
            obs, reward, done, info = env.step(action.item())
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': torch.FloatTensor(log_probs),
            'makespan': obs['makespan'],
        }
    
    def _update_step(self, trajectory: Dict, returns: torch.Tensor) -> float:
        """Perform a single update step"""
        total_loss = 0.0
        
        for tau, state in enumerate(trajectory['states']):
            action = trajectory['actions'][tau]
            old_log_prob = trajectory['log_probs'][tau]
            G = returns[tau]
            
            # Compute new log prob
            logits = self.model(
                state['graph'].x,
                state['graph'].edge_index,
                state['graph'].edge_attr,
                state['graph'].node_types,
                state['agent_idx'],
                torch.LongTensor(state['feasible_tasks']),
            )
            
            probs = F.softmax(logits, dim=0)
            action_dist = torch.distributions.Categorical(probs)
            new_log_prob = action_dist.log_prob(torch.tensor(action))
            
            # Compute ratio
            ratio = torch.exp(new_log_prob - old_log_prob)
            
            # Clip-REINFORCE objective
            clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            loss = -torch.min(ratio * G, clip_ratio * G)
            
            total_loss += loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def _update_baseline(self):
        """Update baseline model with Polyak averaging"""
        for param, baseline_param in zip(
            self.model.parameters(),
            self.baseline_model.parameters()
        ):
            baseline_param.data.copy_(
                self.polyak * baseline_param.data + 
                (1 - self.polyak) * param.data
            )


# ============================================================================
# Training Loop
# ============================================================================

def train_schedulenet(
    instances: List,
    num_epochs: int = 1000,
    save_path: str = "/mnt/user-data/outputs/schedulenet_model.pt",
):
    """
    Main training loop for ScheduleNet
    
    Args:
        instances: List of job shop instances
        num_epochs: Number of training epochs
        save_path: Path to save the trained model
    """
    # Initialize model
    model = ScheduleNet(
        node_feature_dim=5,
        edge_feature_dim=1,
        hidden_dim=32,
        num_layers=2,
        num_node_types=5,
    )
    
    # Initialize trainer
    trainer = ClipREINFORCE(model)
    
    # Training loop
    print("Starting ScheduleNet training...")
    print(f"Number of instances: {len(instances)}")
    print(f"Number of epochs: {num_epochs}")
    print("-" * 60)
    
    all_stats = defaultdict(list)
    
    for epoch in range(num_epochs):
        # Sample random instance
        instance = np.random.choice(instances)
        env = ScheduleNetEnv(instance)
        
        # Train on episode
        stats = trainer.train_episode(env, num_inner_updates=5)
        
        # Track statistics
        for key, value in stats.items():
            all_stats[key].append(value)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            avg_makespan = np.mean(all_stats['makespan'][-10:])
            avg_baseline = np.mean(all_stats['baseline_makespan'][-10:])
            avg_loss = np.mean(all_stats['loss'][-10:])
            
            print(f"Epoch {epoch + 1:4d} | "
                  f"Makespan: {avg_makespan:7.2f} | "
                  f"Baseline: {avg_baseline:7.2f} | "
                  f"Loss: {avg_loss:8.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'stats': dict(all_stats),
            }, save_path)
            print(f"Checkpoint saved to {save_path}")
    
    print("-" * 60)
    print("Training completed!")
    
    # Plot training curves
    plot_training_curves(all_stats, save_path.replace('.pt', '_training.png'))
    
    return model, all_stats


def plot_training_curves(stats: Dict, save_path: str):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Makespan
    axes[0, 0].plot(stats['makespan'], alpha=0.3, label='Policy')
    axes[0, 0].plot(stats['baseline_makespan'], alpha=0.3, label='Baseline')
    
    # Moving average
    window = 50
    if len(stats['makespan']) >= window:
        ma_policy = np.convolve(stats['makespan'], 
                                np.ones(window)/window, mode='valid')
        ma_baseline = np.convolve(stats['baseline_makespan'],
                                  np.ones(window)/window, mode='valid')
        axes[0, 0].plot(ma_policy, linewidth=2, label='Policy (MA)')
        axes[0, 0].plot(ma_baseline, linewidth=2, label='Baseline (MA)')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Makespan')
    axes[0, 0].set_title('Makespan over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(stats['loss'], alpha=0.5)
    if len(stats['loss']) >= window:
        ma_loss = np.convolve(stats['loss'], np.ones(window)/window, mode='valid')
        axes[0, 1].plot(ma_loss, linewidth=2, color='red')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Number of steps
    axes[1, 0].plot(stats['num_steps'], alpha=0.5)
    if len(stats['num_steps']) >= window:
        ma_steps = np.convolve(stats['num_steps'], 
                              np.ones(window)/window, mode='valid')
        axes[1, 0].plot(ma_steps, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Episode Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Improvement over baseline
    improvement = np.array(stats['baseline_makespan']) - np.array(stats['makespan'])
    axes[1, 1].plot(improvement, alpha=0.5)
    if len(improvement) >= window:
        ma_improvement = np.convolve(improvement, 
                                    np.ones(window)/window, mode='valid')
        axes[1, 1].plot(ma_improvement, linewidth=2, color='purple')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Improvement')
    axes[1, 1].set_title('Improvement over Baseline')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


# ============================================================================
# Inference and Evaluation
# ============================================================================

def evaluate_schedulenet(
    model: ScheduleNet,
    instances: List,
    plot_gantt: bool = True,
) -> Dict:
    """
    Evaluate ScheduleNet on test instances
    
    Args:
        model: Trained ScheduleNet model
        instances: List of test instances
        plot_gantt: Whether to plot Gantt charts
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    makespans = []
    
    with torch.no_grad():
        for idx, instance in enumerate(instances):
            env = ScheduleNetEnv(instance)
            obs = env.reset()
            done = False
            
            actions_taken = []
            
            while not done:
                # Get greedy action
                logits = model(
                    obs['graph'].x,
                    obs['graph'].edge_index,
                    obs['graph'].edge_attr,
                    obs['graph'].node_types,
                    obs['agent_idx'],
                    torch.LongTensor(obs['feasible_tasks']),
                )
                
                action = torch.argmax(logits).item()
                actions_taken.append(action)
                
                obs, reward, done, info = env.step(action)
            
            makespan = obs['makespan']
            makespans.append(makespan)
            
            print(f"Instance {idx + 1}: Makespan = {makespan:.2f}")
            
            # Plot Gantt chart for first instance
            if plot_gantt and idx == 0:
                try:
                    plot_gantt_chart(
                        env.env.schedule,
                        title=f"ScheduleNet Solution (Makespan: {makespan:.2f})"
                    )
                    plt.savefig(
                        f"/mnt/user-data/outputs/gantt_chart_instance_{idx}.png",
                        dpi=150,
                        bbox_inches='tight'
                    )
                    plt.close()
                except Exception as e:
                    print(f"Could not plot Gantt chart: {e}")
    
    results = {
        'mean_makespan': np.mean(makespans),
        'std_makespan': np.std(makespans),
        'min_makespan': np.min(makespans),
        'max_makespan': np.max(makespans),
        'all_makespans': makespans,
    }
    
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Mean Makespan: {results['mean_makespan']:.2f} ± {results['std_makespan']:.2f}")
    print(f"  Min Makespan:  {results['min_makespan']:.2f}")
    print(f"  Max Makespan:  {results['max_makespan']:.2f}")
    print("=" * 60)
    
    return results


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # This section shows how to use the code
    # You would replace this with your actual instance generation
    
    print("ScheduleNet Training System")
    print("=" * 60)
    print("To use this code:")
    print("1. Import your generator: from generator import generate_instances")
    print("2. Generate instances: instances = generate_instances()")
    print("3. Train: model, stats = train_schedulenet(instances)")
    print("4. Evaluate: results = evaluate_schedulenet(model, test_instances)")
    print("=" * 60)
