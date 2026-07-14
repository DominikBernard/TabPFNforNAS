import nats_bench
import json
import tqdm
import os

def create_arch_str_to_index_dict(path_to_save: str = "tabpfn_final_accuracy/data/arch_str_to_index.json"):
    """Create a dictionary mapping architecture strings to their corresponding indices in the NATS-Bench search space."""
    print(f"Creating architecture string to index dictionary and saving to {path_to_save}...")
    d = dict()
    api = nats_bench.create(None, 'tss', fast_mode=True, verbose=True)
    for i in tqdm.tqdm(range(15625)):
        arch_str = api.arch(i)
        d[arch_str] = i

    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

    # with indent = 4
    with open(path_to_save, "w") as f:
        json.dump(d, f, indent=4)
    print(f"Completed, dictionary saved to {path_to_save}.")

def create_index_to_arch_str_dict_if_not_exists(path_to_save: str = "tabpfn_final_accuracy/data/arch_str_to_index.json") -> dict[str, int]:
    """
    Create the architecture string to index dictionary if it does not already exist.
    
    Args:
        path_to_save: Path where the dictionary will be saved or loaded from.

    Returns:
        dict: A dictionary mapping architecture strings to their corresponding indices.
    """
    try:
        with open(path_to_save, "r") as f:
            d = json.load(f)
    except FileNotFoundError:
        create_arch_str_to_index_dict(path_to_save)
        with open(path_to_save, "r") as f:
            d = json.load(f)
    return d

if __name__ == "__main__":
    print("Creating architecture string to index dictionary if it does not already exist...")
    create_index_to_arch_str_dict_if_not_exists()
