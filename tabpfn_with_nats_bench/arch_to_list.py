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

if __name__ == "__main__":
    api = create(None, 'tss', fast_mode=True, verbose=True)
    architecture_index = 5
    arch_str = api.arch(architecture_index)
    print(arch_str)
    edges = arch_to_list(arch_str)
    print(edges)
