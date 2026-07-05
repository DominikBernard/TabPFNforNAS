import json

def find_best_n_architecture(data: list[dict], n: int) -> list[tuple[int, float]]:
    """Find the best n architectures in the data."""
    best_archs = sorted([(i, data[i]["testacc"]) for i in range(len(data))], key=lambda x: x[1], reverse=True)[:n]
    return best_archs

if __name__ == "__main__":
    with open("tabpfn_final_accuracy/data/merged_data.json", "r") as f:
        data = json.load(f)

    best_archs = find_best_n_architecture(data, 50)
    print("Best architectures:")
    for i in range(len(best_archs)):
        print(f"{i+1}. Arch index: {best_archs[i][0]}, Test accuracy: {best_archs[i][1]}")

