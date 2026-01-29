import subprocess

# File paths
input_path = "rough_prompt.txt"
template_path = "prompt_template.txt"
output_path = "improved_prompt.txt"

# Read rough prompt
with open(input_path, "r") as f:
    rough_prompt = f.read()

# Read instruction template
with open(template_path, "r") as f:
    template = f.read()

# Combine into final input
final_prompt = template.replace("{{PROMPT}}", rough_prompt)

# Run Gemini CLI with the prompt
cmd = ["gemini", "--model", "gemini-1.5-pro", "--multiline"]

process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
output, _ = process.communicate(input=final_prompt)

# Save output
with open(output_path, "w") as f:
    f.write(output.strip())

print("✅ Optimized prompt saved to", output_path)
