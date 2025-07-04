#!/usr/bin/env python3
"""
generate_ordered.py - Sinh mật khẩu theo pattern với thứ tự xác suất giảm dần

Mục đích:
    - Input: pattern (ví dụ "L4N3S1")
    - Output: file TXT với mật khẩu theo xác suất cao nhất trước
"""

import argparse
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel
from sopg_search.node import Node
from sopg_search.frontier import FrontierQueue
from sopg_search.packing import pack_nodes
from tokenizer.char_tokenizer import CharTokenizer
import random
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def expand_node(node, model, tokenizer, top_k):
    """
    Sinh các child node từ 1 node hiện tại.
    - Query GPT để lấy phân phối token tiếp theo
    - Lọc token theo pattern constraint
    - Chọn top-k token
    - Tất cả trên GPU nếu có
    """
    # ✅ Đưa prefix lên GPU
    model_input = torch.tensor([node.prefix_tokens], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        outputs = model(model_input)
        logits = outputs.logits[:, -1, :]

        # ✅ Tính softmax và log trên GPU luôn
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)

    log_probs = log_probs.squeeze(0)  # Shape [vocab_size], vẫn trên GPU

    # ✅ Lọc theo pattern constraint
    allowed_types = node.allowed_token_types()
    filtered_token_ids = []
    filtered_log_probs = []

    for token_id in range(len(tokenizer.encoder)):
        token_type = tokenizer.get_char_type(token_id)
        if token_type in allowed_types:
            filtered_token_ids.append(token_id)
            filtered_log_probs.append(log_probs[token_id])

    if not filtered_token_ids:
        return []

    # ✅ Chọn top-k
    filtered_log_probs_tensor = torch.tensor(filtered_log_probs, device=DEVICE)
    topk_values, topk_indices = torch.topk(filtered_log_probs_tensor, k=min(top_k, len(filtered_log_probs)))

    children = []
    for idx in topk_indices:
        new_token = filtered_token_ids[idx.item()]
        new_log_prob = filtered_log_probs_tensor[idx].item()
        child = node.copy_with_new_token(new_token, new_log_prob)
        children.append(child)

    return children

def main():
    parser = argparse.ArgumentParser(description="Pattern-Aware SOPG Password Generator")
    parser.add_argument("--pattern", type=str, help="Pattern (e.g., L4N3S1). If not provided, generates random.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned GPT2 model")
    parser.add_argument("--tokenizer_vocab", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--tokenizer_type_map", type=str, required=True, help="Path to char_type_map.json")
    parser.add_argument("--output_txt", type=str, required=True, help="Output TXT file")
    parser.add_argument("--generate_num", type=int, default=10000, help="Number of passwords to generate")
    parser.add_argument("--beam_width", type=int, default=30, help="Beam width (top-k tokens per expansion)")
    parser.add_argument("--packing_threshold", type=float, default=-30.0, help="Log-prob threshold for packing")

    args = parser.parse_args()

    # ✅ Hàm random pattern nhẹ nhàng hơn
    def generate_random_pattern():
        parts = []
        num_L = random.randint(5, 8)
        num_N = random.randint(0, 2)
        num_S = random.randint(0, 1)

        if num_L > 0:
            parts.append(f"L{num_L}")
        if num_N > 0:
            parts.append(f"N{num_N}")
        if num_S > 0:
            parts.append(f"S{num_S}")

        random.shuffle(parts)
        return ''.join(parts)

    if not args.pattern:
        args.pattern = generate_random_pattern()
        print(f"✅ No pattern provided. Generated random pattern: {args.pattern}")

    print("✅ Loading tokenizer...")
    tokenizer = CharTokenizer(
        vocab_file=args.tokenizer_vocab,
        char_type_map_file=args.tokenizer_type_map,
        bos_token="<BOS>",
        eos_token="<EOS>",
        sep_token="<SEP>",
        unk_token="<UNK>",
        pad_token="<PAD>"
    )

    print("✅ Loading model...")
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(DEVICE)
    model.eval()

    print("✅ Initializing frontier...")
    # Parse pattern like "L4N3S1" → {'L': 4, 'N': 3, 'S': 1}
    pattern_state = {}
    matches = re.findall(r'([LNS])(\d+)', args.pattern)
    for t, count in matches:
        pattern_state[t] = int(count)

    initial_prefix = [
        tokenizer.bos_token_id,
        tokenizer._convert_token_to_id(args.pattern),
        tokenizer.sep_token_id
    ]
    root_node = Node(
        prefix_tokens=initial_prefix,
        log_prob=0.0,
        depth=0,
        pattern_state=pattern_state,
        tokenizer=tokenizer,
        pattern=args.pattern
    )

    frontier = FrontierQueue()
    frontier.add(root_node)

    results = []
    pbar = tqdm(total=args.generate_num, desc="Generating passwords")

    max_steps = 100000
    step = 0

    while len(results) < args.generate_num and not frontier.is_empty() and step < max_steps:
        step += 1
        node = frontier.pop()

        if node.is_terminal():
            decoded = tokenizer.decode(torch.tensor(node.prefix_tokens))
            password = decoded.split('<SEP>')[-1].strip()
            results.append((args.pattern, password, node.log_prob))
            pbar.update(1)
            continue

        # Expand node
        children = expand_node(node, model, tokenizer, args.beam_width)
        packed_children = pack_nodes(children, args.packing_threshold)

        for child in packed_children:
            if hasattr(child, 'children'):
                for sub_child in child.children:
                    frontier.add(sub_child)
            else:
                frontier.add(child)

    pbar.close()

    if frontier.is_empty() and len(results) < args.generate_num:
        print(f"⚠️ Frontier cạn kiệt! Chỉ sinh được {len(results)} mật khẩu cho pattern: {args.pattern}")

    print(f"✅ Saving {len(results)} passwords to {args.output_txt}...")

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for _, password, log_prob in results:
            f.write(f"{password}\t{log_prob}\n")

    print("✅ Done.")


if __name__ == "__main__":
    main()
