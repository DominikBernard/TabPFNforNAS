import numpy as np
import regex as re
from nats_bench import create

def arch_to_list(arch_str: str) -> list[int]:
    """Convert architecture string to a list of integers representing edges using regex."""
    edge_mapping = {
        'none': 0,
        'skip_connect': 1,
        'nor_conv_1x1': 2,
        'nor_conv_3x3': 3,
        'avg_pool_3x3': 4
    }
    # Find all edge types using regex pattern: edge~node
    matches = re.findall(r'([a-z_0-9]+)~\d+', arch_str)
    edges = [edge_mapping[match] for match in matches if match in edge_mapping]
    return edges
    
def is_arch_connected(arch_str: str) -> bool:
    """Check if the architecture is connected from input to output."""
    # Extract all nodes and edges using regex
    edges = re.findall(r'([a-z_0-9]+)~(\d+)', arch_str)

    # edges:
    e0_1, e0_2, e1_2, e0_3, e1_3, e2_3 = edges

    # possible paths:
    path1 = (e0_1[0] != 'none') and (e1_2[0] != 'none') and (e2_3[0] != 'none')
    path2 = (e0_1[0] != 'none') and (e1_3[0] != 'none')
    path3 = (e0_2[0] != 'none') and (e2_3[0] != 'none')
    path4 = (e0_3[0] != 'none')

    return path1 or path2 or path3 or path4

def log_scale_synflow(synflow_value: float) -> float:
    """Log scale the synflow value."""
    return np.log10(synflow_value + 1e-12)

def get_features_and_label(data: list[dict], arch_index: int, connected_feature: bool, arch_index_feature: bool, edge_features_onehot: bool, zero_cost_features: list[str]) -> tuple[list[int], float]:
    """Get features and label from architecture data."""
    features = []
    if arch_index_feature:
        features.append(arch_index / 15624)  # normalize index to [0, 1]
    edges = arch_to_list(data[arch_index]["arch"])
    if edge_features_onehot:
        one_hot_edges = []
        for edge in edges:
            one_hot = [0] * 5
            one_hot[edge] = 1
            one_hot_edges.extend(one_hot)
        features.extend(one_hot_edges)
    else:
        features.extend(edges)
    if connected_feature:
        features.append(int(is_arch_connected(data[arch_index]["arch"])))
    if zero_cost_features is not []:
        zero_costs = data[arch_index]["logmeasures"]
        if "f" in zero_cost_features:
            features.append(zero_costs["fisher"])
        if "g_n" in zero_cost_features:
            features.append(zero_costs["grad_norm"])
        if "g" in zero_cost_features:
            features.append(zero_costs["grasp"])
        if "j_c" in zero_cost_features:
            features.append(zero_costs["jacob_cov"])
        if "p" in zero_cost_features:
            features.append(zero_costs["plain"])
        if "s" in zero_cost_features:
            features.append(zero_costs["snip"])
        if "sy" in zero_cost_features:
            features.append(log_scale_synflow(zero_costs["synflow"]))
        if "sy_bn" in zero_cost_features:
            features.append(zero_costs["synflow_bn"])

    label = data[arch_index]["testacc"]
    return features, label

def get_features(arch_str: str, arch_index: int, connected_feature: bool, arch_index_feature: bool, edge_features_onehot: bool) -> list[int]:
    """Get features from architecture string."""
    features = []
    if arch_index_feature:
        features.append(arch_index / 15624)  # normalize index to [0, 1]
    edges = arch_to_list(arch_str)
    if edge_features_onehot:
        one_hot_edges = []
        for edge in edges:
            one_hot = [0] * 5
            one_hot[edge] = 1
            one_hot_edges.extend(one_hot)
        features.extend(one_hot_edges)
    else:
        features.extend(edges)
    if connected_feature:
        features.append(int(is_arch_connected(arch_str)))
    return features

if __name__ == "__main__":
    api = create(None, 'tss', fast_mode=True, verbose=True)
    architecture_index = 5
    arch_str = api.arch(architecture_index)
    print(arch_str)
    edges = arch_to_list(arch_str)
    print(edges)
