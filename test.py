from datasets import load_dataset

configs = ["rc", "rc.nocontext", "unfiltered", "unfiltered.nocontext"]

for cfg in configs:
    ds = load_dataset("trivia_qa", cfg, split='train[:1000]')
    print(f"\n=== Config: {cfg} ===")
    print(f"{len(ds)} questions")
    # for split in ds.keys():
    #     print(f"{split}: {len(ds[split])} questions")
