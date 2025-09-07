import torch


def reconstruct_dense(edge_index: torch.Tensor, edge_attr: torch.Tensor, device: torch.device) -> torch.Tensor:
    """reconstruct a dense adjacency matrix from (edge_index, edge_attr).

    - edge_index: (2, E) long tensor
    - edge_attr: (E,) or (K, E) where second row can hold weights
    - returns a dense [N, N] tensor on the given device
    """
    if edge_index.numel() == 0:
        # empty graph fallback
        return torch.eye(2, device=device)

    num_nodes = int(edge_index.max().item()) + 1
    dtype = edge_attr.dtype if torch.is_tensor(edge_attr) else torch.float32
    dense = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    weights = edge_attr[1] if (edge_attr.dim() > 1 and edge_attr.size(0) >= 2) else edge_attr
    dense[edge_index[0], edge_index[1]] = weights
    return dense


def to_networkx(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    """build a networkx graph from (edge_index, edge_attr).

    done on cpu to avoid device deps; intended for analysis/visualization only.
    """
    import networkx as nx

    G = nx.Graph()
    if edge_index.numel() == 0:
        return G
    edges_cpu = edge_index.cpu().numpy()
    if torch.is_tensor(edge_attr):
        weights_cpu = edge_attr[1].cpu().numpy() if edge_attr.dim() > 1 else edge_attr.cpu().numpy()
    else:
        weights_cpu = None
    for i in range(edges_cpu.shape[1]):
        src, dst = int(edges_cpu[0, i]), int(edges_cpu[1, i])
        if weights_cpu is None:
            G.add_edge(src, dst, weight=1.0)
        else:
            w = float(weights_cpu[i]) if getattr(weights_cpu, 'ndim', 1) > 0 else 1.0
            G.add_edge(src, dst, weight=w)
    return G


