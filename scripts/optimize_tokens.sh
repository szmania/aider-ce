#!/bin/bash

echo "Calculating Ideal Tokens For Tokenization"

# Write the Python script to a temporary file
cat << 'EOF' > /tmp/table_optimizer.py
import string
import pprint
import tiktoken
from transformers import AutoTokenizer

def generate_universal_prefix_free_tokens(target_count=256):
    print("Loading tokenizers... (This will download vocab files to your cache)")
    
    # 1. OpenAI Baseline
    enc_openai = tiktoken.get_encoding("cl100k_base")
    
    # 2. Stable Open-Weight Tokenizers (Publicly Accessible)
    tokenizers = {
        "mistral": AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"),
        "qwen": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"),
        "deepseek": AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
    }
    
    # Base36: 0-9, a-z (36 characters total)
    base36 = list(string.digits + string.ascii_lowercase)
    
    direct_chars = base36.copy()
    prefix_pairs = []

    def is_single_token_everywhere(text):
        if len(enc_openai.encode(text)) != 1:
            return False
        for name, enc in tokenizers.items():
            # add_special_tokens=False is critical for correct subword counting
            if len(enc.encode(text, add_special_tokens=False)) != 1:
                return False
        return True

    print("Calculating intersection of all tokenizers...")
    
    while len(direct_chars) + len(prefix_pairs) < target_count:
        if not direct_chars:
            raise ValueError("Stricter rules exhausted the alphabet. Please switch to Base62 (A-Z, a-z, 0-9).")
            
        prefix = direct_chars.pop() 
        for second_char in base36:
            seq = prefix + second_char
            if is_single_token_everywhere(seq):
                prefix_pairs.append(seq)
                
    return (direct_chars + prefix_pairs)[:target_count]

if __name__ == "__main__":
    try:
        optimal_tokens = generate_universal_prefix_free_tokens(256)
        print("\nBASE256_UNIVERSAL_CODES = ", end="")
        pprint.pprint(optimal_tokens, compact=True, width=100)
    except Exception as e:
        print(f"Error: {e}")
EOF

# Execute the python script with the bash arguments
python3 /tmp/table_optimizer.py

# Clean up
rm /tmp/table_optimizer.py