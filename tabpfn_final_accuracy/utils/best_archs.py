import json

from tabpfn_final_accuracy.utils.dataloader import download_data_if_not_exists

def find_best_n_architecture(data: list[dict], n: int) -> list[tuple[int, float]]:
    """Find the best n architectures in the data."""
    best_archs = sorted([(i, data[i]["testacc"]) for i in range(len(data))], key=lambda x: x[1], reverse=True)[:n]
    return best_archs

if __name__ == "__main__":
    path_to_data = "tabpfn_final_accuracy/data/merged_data.json"

    download_data_if_not_exists(path_to_data)

    with open(path_to_data, "r") as f:
        data = json.load(f)

    n = 10

    best_archs = find_best_n_architecture(data, n)
    print("Best architectures:")
    for i in range(len(best_archs)):
        print(f"{i+1}. Arch index: {best_archs[i][0]}, Test accuracy: {best_archs[i][1]}")
    print(f"Average test accuracy of best {n} architectures: {sum([arch[1] for arch in best_archs]) / len(best_archs):.4f}")

