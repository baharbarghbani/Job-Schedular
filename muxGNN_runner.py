import generator
import networkx as nx
import dgl
import torch
import numpy as np
from muxGNN.model.mux_gnn import MuxGNN

def build_task_dag_graph(instance):
    """
    Build task DAG graph from job shop instance.
    
    Args:
        instance: JobShopInstance with jobs containing operations
        
    Returns:
        DGL graph with task nodes and dependency edges
    """
    # Create task list with job and operation IDs
    tasks = []
    task_id = 0
    job_to_tasks = {}
    
    for job_id, job in enumerate(instance.jobs):
        job_to_tasks[job_id] = []
        for op_id, operation in enumerate(job):
            tasks.append({
                'task_id': task_id,
                'job_id': job_id,
                'op_id': op_id,
                'duration': operation.duration,
                'machines': [operation.machines] if isinstance(operation.machines, int) else operation.machines
            })
            job_to_tasks[job_id].append(task_id)
            task_id += 1
    
    # Create graph with task dependencies
    G = nx.DiGraph()
    num_tasks = len(tasks)
    
    # Add nodes
    for i in range(num_tasks):
        G.add_node(i)
    
    # Add edges for job precedence constraints
    for job_id, task_ids in job_to_tasks.items():
        for i in range(len(task_ids) - 1):
            G.add_edge(task_ids[i], task_ids[i + 1])
    
    return G, tasks

def build_machine_graphs(instance, tasks):
    """
    Build machine-specific graphs where nodes are tasks executable on that machine.
    
    Args:
        instance: JobShopInstance
        tasks: List of task dictionaries
        
    Returns:
        Dictionary mapping machine_id to machine-specific graph
    """
    machine_graphs = {}
    num_machines = max([machine for task in tasks for machine in task['machines']]) + 1
    
    for machine_id in range(num_machines):
        G = nx.DiGraph()
        
        # Add nodes for tasks that can run on this machine
        machine_tasks = [i for i, task in enumerate(tasks) if machine_id in task['machines']]
        for task_id in machine_tasks:
            G.add_node(task_id)
        
        # Add undirected edges between tasks that can run on the same machine (potential conflicts)
        for i in machine_tasks:
            for j in machine_tasks:
                if i < j:   
                    G.add_edge(i, j)
                    G.add_edge(j, i)
        
        machine_graphs[machine_id] = G
    
    return machine_graphs

def extract_node_features(tasks, num_tasks):
    """
    Extract node features for tasks.
    
    Args:
        tasks: List of task dictionaries
        num_tasks: Total number of tasks
        
    Returns:
        Tensor of shape (num_tasks, feat_dim)
    """
    features = []
    for i in range(num_tasks):
        task = tasks[i]
        # Feature: [duration, job_id (normalized), op_id (normalized), num_machines]
        feat = [
            task['duration'] / 100.0,  # Normalize duration
            task['job_id'] / max(1, len(set(t['job_id'] for t in tasks))),
            task['op_id'] / max(1, max(t['op_id'] for t in tasks)),
            len(task['machines']) / max(1, len(set(m for t in tasks for m in t['machines'])))
        ]
        features.append(feat)
    
    return torch.tensor(features, dtype=torch.float32)

def graphs_to_dgl(task_dag, machine_graphs, features):
    """
    Convert NetworkX graphs to DGL graphs with features.
    
    Args:
        task_dag: NetworkX DiGraph for task DAG
        machine_graphs: Dict of machine_id -> NetworkX DiGraph
        features: Node features tensor
        
    Returns:
        List of DGL graphs (task DAG followed by per-machine graphs)
    """
    dgl_graphs = []
    
    # Convert task DAG
    task_dag_dgl = dgl.from_networkx(task_dag)
    task_dag_dgl.ndata['feat'] = features
    dgl_graphs.append(task_dag_dgl)
    
    # Convert machine graphs
    for machine_id in sorted(machine_graphs.keys()):
        G = machine_graphs[machine_id]
        if len(G) > 0:
            machine_graph_dgl = dgl.from_networkx(G)
            machine_graph_dgl.ndata['feat'] = features
            dgl_graphs.append(machine_graph_dgl)
    
    return dgl_graphs

def get_machine_state(instance):
    """
    Extract current machine state (capacity, utilization, etc.).
    
    Args:
        instance: JobShopInstance
        
    Returns:
        Tensor representing machine state
    """
    num_machines = max([op.machines for job in instance.jobs for op in job]) + 1
    # Initialize machine state: [available_capacity, current_load, queue_length] per machine
    machine_state = torch.zeros((num_machines, 3), dtype=torch.float32)
    
    for m in range(num_machines):
        machine_state[m, 0] = 1.0  # available_capacity (normalized to 1.0)
        machine_state[m, 1] = 0.0  # current_load (initially 0)
        machine_state[m, 2] = 0.0  # queue_length (initially 0)
    
    return machine_state



def build_mux_hetero_graph(task_dag, machine_graphs, features):
    """
    Build a multiplex graph representation for MuxGNN.
    Returns graph metadata for later DGL conversion.
    """
    num_nodes = features.shape[0]
    
    # Store relation information
    relations_data = {
        'task': list(task_dag.edges()),
    }
    
    # Machine relations
    for m_id, G in machine_graphs.items():
        if G.number_of_edges() > 0:
            relations_data[f'machine_{m_id}'] = list(G.edges())
    
    return {
        'num_nodes': num_nodes,
        'relations': relations_data,
        'features': features
    }, list(relations_data.keys())



def embed_instance(instance, model=None, device='cpu'):
    """
    Embed a job shop instance.
    
    Currently returns graph and feature information.
    Full MuxGNN+PPO integration pending DGL version fix.
    """
    # --- Build graphs ---
    task_dag, tasks = build_task_dag_graph(instance)
    machine_graphs = build_machine_graphs(instance, tasks)

    num_tasks = len(tasks)

    # --- Features ---
    features = extract_node_features(tasks, num_tasks)

    # --- Heterograph metadata for future MuxGNN ---
    graph_data, relations = build_mux_hetero_graph(
        task_dag,
        machine_graphs,
        features
    )

    # --- Machine state ---
    machine_state = get_machine_state(instance).to(device)

    # Return embedding information
    embedding = {
        'task_dag': task_dag,
        'machine_graphs': machine_graphs,
        'task_embeddings': features.to(device),  # Raw features as embedding until MuxGNN is available
        'machine_state': machine_state,
        'num_tasks': num_tasks,
        'num_machines': machine_state.shape[0],
        'tasks': tasks,
        'graph_data': graph_data,
        'relations': relations
    }
    
    return embedding
    

if __name__ == "__main__":
    instance_list = generator.generate_instances()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for idx, instance in enumerate(instance_list[:5]):  # Process first 5 instances for testing
        embedding = embed_instance(instance, model=None, device=device)
        print(f"\nInstance {idx}:")
        print(f"  Tasks: {embedding['num_tasks']}")
        print(f"  Machines: {embedding['num_machines']}")
        print(f"  Task embeddings shape: {embedding['task_embeddings'].shape}")
        print(f"  Machine state shape: {embedding['machine_state'].shape}")
        print(f"  Task DAG edges: {embedding['task_dag'].number_of_edges()}")
        print(f"  Relations: {embedding['relations']}")