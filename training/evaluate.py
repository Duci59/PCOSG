# This script evaluates password lists: calculates Hit Rate and Repeat Rate
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--test_file", help="Path to test password .txt file", type=str, required=True)
parser.add_argument("--gen_path", help="Path to generated password files", type=str, required=True)
args = parser.parse_args()

# Collect all .txt files under the generated path
def get_all_files(path):
    if os.path.isfile(path) and path.endswith(".txt"):
        return [path]
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".txt"):
                files.append(os.path.join(root, filename))
    return files

gen_files = get_all_files(args.gen_path)

# Extract only password strings (2nd token) from each line in gen files
def get_gen_passwords(gen_files):
    gen_passwords = []
    for gen_file in gen_files:
        with open(gen_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    gen_passwords.append(parts[1].strip())
    
    return gen_passwords
    

# Calculate Hit Rate: how many test passwords are found in generated passwords
def get_hit_rate(test_file, gen_files):
    gen_passwords = set(get_gen_passwords(gen_files))
    with open(test_file, "r") as f:
        test_passwords = set(line.strip() for line in f if line.strip())
    hit_num = sum(1 for pw in gen_passwords if pw in test_passwords)
    hit_rate = hit_num / len(test_passwords) if test_passwords else 0
    return hit_rate

# Calculate Repeat Rate: 1 - (unique / total)
def get_repeat_rate(gen_files):
    gen_passwords = get_gen_passwords(gen_files)
    unique_passwords = set(gen_passwords)
    total = len(gen_passwords)
    repeat_rate = 1 - len(unique_passwords) / total if total else 0
    return repeat_rate

# Run evaluation
hit_rate = get_hit_rate(args.test_file, gen_files)
repeat_rate = get_repeat_rate(gen_files)

# Print results
print("Hit Rate: {:.2f}%".format(hit_rate * 100))
print("Repeat Rate: {:.2f}%".format(repeat_rate * 100))