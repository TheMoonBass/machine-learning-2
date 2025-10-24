import os
import random as rand
from pathlib import Path
import json
import re
from collections import Counter

# Functions
def findAllWords(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

# Configs
DATASET_DIR = Path('20_newsgroups/20_newsgroups')
OUTPUT_SAVE_DIR = Path('indexes')
OUTPUT_SAVE_DIR.mkdir(exist_ok=True)
rand.seed(777)

# Split Data
train_set = {}
test_set = {}

for category_dir in DATASET_DIR.iterdir():
    if category_dir.is_dir():
        files = list(category_dir.glob('*'))

        rand.shuffle(files)
        train_files = files[:500]
        test_files = files[500:1000]

        train_set[category_dir.name] = [
            str(f.relative_to(DATASET_DIR)) for f in train_files
        ]
        test_set[category_dir.name] = [
            str(f.relative_to(DATASET_DIR)) for f in test_files
        ]

# Define file paths and save as JSON for later use
train_index_file = OUTPUT_SAVE_DIR / "training_index.json"
test_index_file = OUTPUT_SAVE_DIR / "testing_index.json"

train_index_file.write_text(json.dumps(train_set, indent=2))
test_index_file.write_text(json.dumps(test_set, indent=2))


with open("indexes/training_index.json") as f:
    training_index = json.load(f)
    category_counts = {category: Counter() for category in training_index}
    total_words = Counter()
    for category, file_list in training_index.items():
        for file_name in file_list:
            file_path = DATASET_DIR / file_name
            