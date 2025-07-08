import re
import argparse

def check_pattern(pattern: str, password: str) -> bool:
    type_map = {
        'L': lambda c: c.isalpha(),
        'N': lambda c: c.isdigit(),
        'S': lambda c: not c.isalnum()
    }

    parts = re.findall(r'([LNS])(\d+)', pattern)
    pos = 0
    for t, length_str in parts:
        length = int(length_str)
        if pos + length > len(password):
            return False
        segment = password[pos:pos + length]
        if not all(type_map[t](c) for c in segment):
            return False
        pos += length
    return pos == len(password)

def check_file(filepath: str, pattern: str):
    total = 0
    match = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            password = parts[1]
            total += 1
            if check_pattern(pattern, password):
                match += 1

    print(f"{filepath} → Pattern {pattern}: {match}/{total} matched → {match / total * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Check password pattern in file")
    parser.add_argument('--file', required=True, help='Path to the input file')
    parser.add_argument('--pattern', required=True, help='Pattern to check (e.g., L4N2S1)')

    args = parser.parse_args()

    check_file(args.file, args.pattern)

if __name__ == "__main__":
    main()
