import numpy as np
import json
import os
import gzip

print("=== Riemann Zeta Zeros 10k Generator ===\n")

# Look for the correct files on Desktop
possible_files = ['zeros1', 'zeros1.gz', 'zeros1.txt']

source_file = None
for fname in possible_files:
    if os.path.exists(fname):
        source_file = fname
        break

if not source_file:
    print("ERROR: zeros1 or zeros1.gz not found on your Desktop!")
    print("\nPlease download it from here:")
    print("https://www-users.cse.umn.edu/~odlyzko/zeta_tables/")
    print("→ Click on 'zeros1.gz' (gzip'd text)")
    print("Then extract it so you have a file named 'zeros1'")
    exit()

print(f"Found: {source_file} — reading the first 10,000 zeros...")

zeros = []
try:
    if source_file.endswith('.gz'):
        with gzip.open(source_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                val = line.strip()
                if val:
                    zeros.append(float(val))
    else:
        with open(source_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                val = line.strip()
                if val:
                    zeros.append(float(val))

except Exception as e:
    print(f"Error while reading the file: {e}")
    print("Make sure the file 'zeros1' is a plain text file with one number per line.")
    exit()

if len(zeros) < 1000:
    print(f"Warning: Only read {len(zeros)} values. The file may be corrupted.")
else:
    # Save as JSON
    with open('zeta_zeros_10k.json', 'w') as f:
        json.dump(zeros, f, indent=2)

    print(f"\n✅ SUCCESS!")
    print(f"Created zeta_zeros_10k.json with {len(zeros):,} zeros.")
    print(f"First zero  : {zeros[0]:.10f}")
    print(f"Last zero   : {zeros[-1]:.6f}")
    print("\nYou can now run your optimal-prime1.py script.")