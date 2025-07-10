import re

def pattern_to_regex(pattern):
    """
    Convert a pattern like 'L3N5S2' to a regex.
    E.g., L3N5 -> [a-zA-Z]{3}[0-9]{5}
    """
    parts = re.findall(r'[LNS]\d+', pattern)
    regex_parts = []
    for part in parts:
        type_char = part[0]
        count = int(part[1:])
        if type_char == 'L':
            regex_parts.append(f'[a-zA-Z]{{{count}}}')
        elif type_char == 'N':
            regex_parts.append(f'[0-9]{{{count}}}')
        elif type_char == 'S':
            regex_parts.append(f'[^a-zA-Z0-9]{{{count}}}')
    return '^' + ''.join(regex_parts) + '$'

def filter_passwords(input_file, output_file, pattern):
    regex_string = pattern_to_regex(pattern)
    print(f'Compiled regex: {regex_string}')
    regex = re.compile(regex_string)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    matched = [pw for pw in lines if regex.match(pw)]

    with open(output_file, 'w', encoding='utf-8') as f:
        for pw in matched:
            f.write(pw + '\n')

    print(f'Found {len(matched)} password(s) matching pattern "{pattern}".')

if __name__ == '__main__':
    input_file = '../dataset/rockyou-cleaned-Test.txt'
    output_file = 'filtered_passwords.txt'
    
    pattern = 'L6'
    
    filter_passwords(input_file, output_file, pattern)
