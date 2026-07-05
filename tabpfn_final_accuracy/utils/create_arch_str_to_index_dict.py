import nats_bench
import json

d = dict()
api = nats_bench.create(None, 'tss', fast_mode=True, verbose=True)
for i in range(15625):
    arch_str = api.arch(i)
    d[arch_str] = i

# with indent = 4
with open("tabpfn_final_accuracy/data/arch_str_to_index.json", "w") as f:
    json.dump(d, f, indent=4)


