from pathlib import Path
import json
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

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
        prob *= (p_xi_given_y * 10000)
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
    results = []
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
            results.append((category, best_category))

    total_correct = 0
    for cat, pred in results:
        if cat == pred:
            total_correct += 1
    
    total_results = len(results)
    total_acc = total_correct / total_results
    print(f'Total Accuracy: {total_acc:.2%}')

    true_labels = [t for t, _ in results]
    pred_labels = [p for _, p in results]
    cats = sorted(set(true_labels) | set(pred_labels))

    cm = confusion_matrix(true_labels, pred_labels, labels=cats)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=cats, yticklabels=cats,
                cmap='Blues', annot=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()