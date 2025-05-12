import os
import re

input_root = "gemma-4b-resultbase"
output_file = "gemma-4b-resultbase_accuracy_results.txt"
gold_answer = "A"  # <-- You set this manually

def extract_model_answer(line):
    match = re.search(r"Qwen2\.5-32B result:\s*([A-D])", line)
    #match = re.search(r"Gemma-4B result:\s*([A-D])", line)
    return match.group(1) if match else None

results = []

for root, _, files in os.walk(input_root):
    for file in files:
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(root, file)

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total = 0
        correct = 0
        for line in lines:
            pred = extract_model_answer(line)
            if pred:
                total += 1
                if pred.strip() == gold_answer:
                    correct += 1

        acc = correct / total if total > 0 else 0.0
        results.append((file_path, acc, correct, total))

# Write all results to output file
with open(output_file, 'w', encoding='utf-8') as out:
    for path, acc, correct, total in results:
        out.write(f"{path}: ACC={acc:.2%} ({correct}/{total})\n")

print(f"Done! Accuracy results saved to {output_file}")
