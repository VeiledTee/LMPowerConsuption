import json
from datasets import load_dataset
from bs4 import BeautifulSoup
from tqdm import tqdm


def extract_first_paragraph(html):
    soup = BeautifulSoup(html or "", "html.parser")
    p = soup.find("p")
    return p.get_text().strip() if p else ""


# Load your subset
subset = []
subset_by_id = {}
subset_by_q = {}

with open(r"C:\Users\Ethan\Documents\PhD\LMPowerConsuption\data\nq_mini_1000.jsonl", "r", encoding="utf-8") as f:
    for ln in f:
        obj = json.loads(ln)
        subset.append(obj)
        subset_by_id[obj["id"]] = obj

subset_ids = set(subset_by_id.keys())

# Load HF dataset
nq = load_dataset("google-research-datasets/natural_questions", "default")["validation"]


out_path = r"C:\Users\Ethan\Documents\PhD\LMPowerConsuption\data\nq_complete_mini_1000.jsonl"
print(len(subset_ids))
with open(out_path, "w", encoding="utf-8") as out_f:
    pbar = tqdm(total=len(nq), desc="Processing NQ dataset")
    matches = 0

    for row in nq:
        rid = row["id"]
        if rid in subset_ids:
            matches += 1
            # Update description with match count
            pbar.set_description(f"Processing NQ dataset (matches: {matches})")

            original_obj = subset_by_id[rid]
            html = row["document"]["html"]
            first_p = extract_first_paragraph(html)

            merged = dict(original_obj)
            merged["html"] = html
            merged["first_paragraph"] = first_p

            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")

        pbar.update(1)

    pbar.close()
