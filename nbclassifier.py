import os
import random as rand
from pathlib import Path
import json
import re
from collections import Counter, defaultdict

# Functions
def findAllWords(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

def calcConditionalProbs(words: list, counts: Counter):
    total_word_counts: int = sum(counts.values())
    prob: float = 1.0
    for word in words:
        word_count = counts[word]
        if word_count > 0:
            p_xi_given_y = word_count / total_word_counts
        else:
            p_xi_given_y = 1 / total_word_counts
        prob *= p_xi_given_y
    return prob

# Configs
DATASET_DIR = Path('20_newsgroups/20_newsgroups')

# Training
with open("indexes/training_index.json") as f:
    training_index = json.load(f)
    category_counts = {category: Counter() for category in training_index}
    total_words = Counter()
    for category, file_list in training_index.items():
        for file_name in file_list:
            file_path = DATASET_DIR / file_name
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            words = findAllWords(text)

            category_counts[category].update(words)
            total_words.update(words)
    
    word_category_counts = defaultdict(set)

    for category, counter in category_counts.items():
        for word, count in counter.items():
            if count > 0:
                word_category_counts[word].add(category)

    # Filter words per category
    filtered_category_word_counts = {}

    for category, counter in category_counts.items():
        filtered_counter = Counter()
        for word, count in counter.items():
            # Keep if frequent enough in this category OR unique to this category
            if count >= 10 or len(word_category_counts[word]) == 1:
                filtered_counter[word] = count
        filtered_category_word_counts[category] = filtered_counter

    # Replace old dictionary
    category_counts = filtered_category_word_counts

with open('indexes/testing_index.json') as f2:
    test_index = json.load(f2)
    categories = category_counts.keys()
    for category, file_list in test_index.items():
        for file_name in file_list:
            best_category = 'NONE'
            best_category_prob = float('-inf')
            file_path = DATASET_DIR / file_name
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            words = findAllWords(text)

            prior = 0.05

            for pot_category in categories:
                product = calcConditionalProbs(words, category_counts[pot_category])
                total_prob = prior * product
                if total_prob > best_category_prob:
                    best_category_prob = total_prob
                    best_category = pot_category
            print(f'File is: {file_name} -- Predicted Category: {best_category} with probability = {best_category_prob} -- True Category: {category}')

