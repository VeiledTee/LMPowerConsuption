import json
from datasets import load_dataset
from tqdm import tqdm

jsonl_path = (
    "/home/penguins/Documents/LMPowerConsumption/data/hotpot_qa_mini_1000.jsonl"
)
out_jsonl = (
    "/home/penguins/Documents/LMPowerConsumption/data/hotpot_qa_mini_1000_indexed.jsonl"
)

# --- load jsonl subset into list and map qid -> object ---
subset = []
id_map = {}
with open(jsonl_path, "r") as f:
    for line in f:
        ex = json.loads(line)
        qid = ex.get("_id") or ex.get("id")
        if qid is None:
            continue
        # initialize field so it's present even if not matched
        ex["qid"] = None
        subset.append(ex)
        id_map[qid] = ex

print(f"Loaded {len(subset)} jsonl entries; mapping contains {len(id_map)} ids")

# --- load HotpotQA fullwiki validation ---
hotpot = load_dataset("hotpot_qa", "fullwiki", split="validation")
print(f"HotpotQA validation size: {len(hotpot)}")

# --- iterate and annotate matching jsonl entries ---
matched_indices = []
for idx, example in tqdm(enumerate(hotpot), total=len(hotpot), desc="Matching qids"):
    qid = example.get("_id") or example.get("id")
    if qid is None:
        continue
    if qid in id_map:
        id_map[qid]["qid"] = idx  # add the index to the jsonl object
        matched_indices.append(idx)

print(f"Matched {len(matched_indices)} instances")
print(matched_indices)

# --- save updated jsonl (preserves original order of subset) ---
with open(out_jsonl, "w") as f:
    for ex in subset:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Saved annotated jsonl -> {out_jsonl}")
