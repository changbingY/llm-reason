import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define input and output paths
input_root = "multiple/"
output_root = "gemma3-4B-result+kp+trans/"
os.makedirs(output_root, exist_ok=True)

# Set your Hugging Face token
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

# Language folder names
languages = {
    "Fwe", "Gyeli", "Ik", "Japhug", "Kagayanen", "Kalamang", "Komnzo",
    "Mauwake", "Mehweb", "Moloko", "Palula", "Papuan_Malay", "Pichi",
    "Rapa_Nui", "Tuatschin", "Ulwa", "Vamale", "Yauyos_Quecha"
}

# Load model and tokenizer
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=True
)

# Helper: extract language from path
def extract_language(filepath):
    parts = Path(filepath).parts
    for part in parts:
        if part in languages:
            return part
    return "unknown"

# Loop through all files
for root, _, files in os.walk(input_root):
    for file in files:
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(root, file)
        language = extract_language(file_path)
        if language == "unknown":
            continue

        # Output setup
        lang_output_dir = os.path.join(output_root, language)
        os.makedirs(lang_output_dir, exist_ok=True)
        file_stem = Path(file).stem
        output_filename = f"{file_stem}_result.txt"
        output_path = os.path.join(lang_output_dir, output_filename)

        with open(file_path, 'r') as f_in, open(output_path, 'w') as f_out:
            lines = f_in.readlines()
            for idx, line in enumerate(lines):
                if 'Question ' in line:
                    f_out.write(line)
                    prompt = (
                        lines[idx+1] + lines[idx+2] + lines[idx+3] +
                        lines[idx+4] + lines[idx+5] +
                        lines[idx+6] + lines[idx+7] + lines[idx+8] +
                        lines[idx+9] + lines[idx+10]
                    )
                    print(f"\nPrompt from {file_path}:\n", prompt)
                    f_out.write(prompt)

                    try:
                        with torch.no_grad():
                            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                            outputs = model.generate(
                                **inputs,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                max_new_tokens=512,
                                eos_token_id=tokenizer.eos_token_id
                            )
                            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    except Exception as e:
                        response = f"[ERROR during generation: {e}]"

                    print("Model response:\n", response)
                    f_out.write('Gemma3-4B result: ' + response + '\n')
