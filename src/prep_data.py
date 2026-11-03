import json
import random
from pathlib import Path
from datasets import load_dataset

DANGEROUS_PATTERNS = [
    "rm -rf",
    "sudo",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    ":(){",
    "chmod -R 777 /",
    "> /dev/sd"
]

def contains_dangerous(cmd: str) -> bool:
    for pattern in DANGEROUS_PATTERNS:
        if pattern in cmd:
            return True
    return False

def make_messages(nl: str, bash: str = None):
    msgs = [
        {"role": "system", "content": "Convert the request into one Bash command. Output only the command."},
        {"role": "user", "content": nl}
    ]
    if bash is not None:
        msgs.append({"role": "assistant", "content": bash})
    return msgs

def main():
    random.seed(42)
    output_dir = Path("/home/shivsharan/nl2shell-project/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HuggingFace NL2SH-ALFA dataset...")
    ds_train = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
    ds_test = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")

    raw_train_count = len(ds_train)
    filtered_train = []
    removed_dangerous_count = 0

    seen_pairs = set()
    for item in ds_train:
        nl = item["nl"].strip()
        bash = item["bash"].strip()
        if not nl or not bash:
            continue
        if contains_dangerous(bash) or contains_dangerous(nl):
            removed_dangerous_count += 1
            continue
        
        # Deduplication by (nl, bash)
        pair = (nl, bash)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        filtered_train.append({"nl": nl, "bash": bash})

    print(f"Total raw train: {raw_train_count}")
    print(f"Removed dangerous examples: {removed_dangerous_count}")
    print(f"Unique safe train examples: {len(filtered_train)}")

    # Shuffle with seed 42
    random.shuffle(filtered_train)

    train_data = filtered_train[:1000]
    val_data = filtered_train[1000:1100]

    # Test data: full 300 tasks
    test_data = []
    for item in ds_test:
        test_data.append({
            "nl": item["nl"].strip(),
            "bash": item["bash"].strip(),
            "bash2": item.get("bash2", "").strip() if item.get("bash2") else "",
            "difficulty": item.get("difficulty", 0),
            "messages": make_messages(item["nl"].strip(), item["bash"].strip())
        })

    # Add messages to train and val
    for item in train_data:
        item["messages"] = make_messages(item["nl"], item["bash"])
    for item in val_data:
        item["messages"] = make_messages(item["nl"], item["bash"])

    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    test_file = output_dir / "test.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(train_data)} train samples to {train_file}")
    print(f"Saved {len(val_data)} val samples to {val_file}")
    print(f"Saved {len(test_data)} test samples to {test_file}")

    # Log summary stats
    summary = {
        "raw_train": raw_train_count,
        "removed_dangerous": removed_dangerous_count,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data)
    }
    with open(output_dir / "data_prep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
