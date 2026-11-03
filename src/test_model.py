from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

messages = [
    {"role": "system", "content": "Convert the request into one Bash command. Output only the command."},
    {"role": "user", "content": "list files in current directory"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Rendered prompt:")
print(repr(text))

inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
print("Generated command:", repr(gen_text))
print("Peak VRAM:", torch.cuda.max_memory_allocated() / (1024**2), "MB")

