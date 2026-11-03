import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

class ShellDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 160):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        nl = item["nl"]
        bash = item["bash"]

        user_messages = [
            {"role": "system", "content": "Convert the request into one Bash command. Output only the command."},
            {"role": "user", "content": nl}
        ]
        full_messages = user_messages + [{"role": "assistant", "content": bash}]

        prompt_res = self.tokenizer.apply_chat_template(user_messages, tokenize=True, add_generation_prompt=True)
        full_res = self.tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)

        prompt_ids = list(prompt_res["input_ids"]) if isinstance(prompt_res, dict) or hasattr(prompt_res, "keys") else list(prompt_res)
        full_ids = list(full_res["input_ids"]) if isinstance(full_res, dict) or hasattr(full_res, "keys") else list(full_res)

        # Append eos_token_id if not present
        if len(full_ids) == 0 or full_ids[-1] != self.tokenizer.eos_token_id:
            full_ids.append(self.tokenizer.eos_token_id)

        # Truncate
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Loss mask: -100 for prompt tokens
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_masks = []
    labels = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_masks.append(torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels)
    }

def get_dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)

def train(args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"=== Training {args.method.upper()} on {model_id} ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tb_dir = Path("/home/shivsharan/nl2shell-project/logs") / f"tb_{args.method}"
    writer = SummaryWriter(log_dir=str(tb_dir))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    if args.method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda",
            torch_dtype=torch.float16
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else: # lora
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.float16
        )

    base_model.gradient_checkpointing_enable()
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    train_dataset = ShellDataset("/home/shivsharan/nl2shell-project/data/train.jsonl", tokenizer, args.max_seq_len)
    val_dataset = ShellDataset("/home/shivsharan/nl2shell-project/data/val.jsonl", tokenizer, args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    step = 0
    epoch_losses = []
    val_losses = []

    model.train()
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        running_loss = 0.0
        count = 0
        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += outputs.loss.item()
            count += 1

            if (b_idx + 1) % args.grad_accum == 0 or (b_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                writer.add_scalar("Loss/train_step", outputs.loss.item(), step)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], step)

        avg_train_loss = running_loss / count
        epoch_losses.append(avg_train_loss)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch + 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += out.loss.item()
                val_count += 1
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        val_losses.append(avg_val_loss)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch + 1)
        model.train()

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    elapsed_time = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Save adapter
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    adapter_size_mb = get_dir_size_mb(out_path)
    writer.close()

    metrics = {
        "method": args.method,
        "epochs": args.epochs,
        "train_time_sec": elapsed_time,
        "peak_vram_mb": peak_vram_mb,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "adapter_size_mb": adapter_size_mb,
        "final_train_loss": epoch_losses[-1],
        "final_val_loss": val_losses[-1],
        "loss_history": epoch_losses,
        "val_history": val_losses
    }
    with open(out_path / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done training {args.method} in {elapsed_time:.1f}s. Peak VRAM: {peak_vram_mb:.1f} MB, Adapter size: {adapter_size_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "qlora"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=160)
    args = parser.parse_args()
    train(args)
