"""
Before preprocessing: split-only baselines.

Word Recall and Count Vectors + Cosine are scored in this file. Both use
text.split() only. Neither calls WordRecallThresholdFactChecker, so later
preprocessing inside factcheck.py does not change these results.

Each printed accuracy is a development-set-tuned baseline accuracy: the
threshold is chosen on the same labeled development facts that are scored.
"""

import math
from collections import Counter
from pathlib import Path

from factchecking_main import read_fact_examples, read_passages


def word_recall_score(fact, passages):
    """
    Maximum word recall of the fact against any passage.
    Tokens are the raw result of text.split() with no other preprocessing.
    Returns None when the fact has no tokens or no passages are supplied.
    Gold labels are not used.
    """
    fact_tokens = set(fact.split())
    if len(fact_tokens) == 0 or not passages:
        return None
    best_score = 0.0
    for passage in passages:
        passage_tokens = set(passage["text"].split())
        score = len(fact_tokens & passage_tokens) / len(fact_tokens)
        if score > best_score:
            best_score = score
    return best_score


def predict_word_recall(fact, passages, threshold):
    """Return S or NS. An empty fact or no passages is always NS."""
    score = word_recall_score(fact, passages)
    if score is None:
        return "NS"
    if score >= threshold:
        return "S"
    return "NS"


def cosine_similarity(fact_counts, passage_counts):
    """
    Cosine of two count vectors. Each unique token is one shared dimension.
    A token missing from one text has count zero, and that zero is included
    in the norm. A zero vector has cosine 0. Gold labels are not used.
    """
    vocabulary = set(fact_counts) | set(passage_counts)
    dot = 0.0
    fact_square = 0.0
    passage_square = 0.0
    for token in vocabulary:
        fact_count = fact_counts[token]
        passage_count = passage_counts[token]
        dot += fact_count * passage_count
        fact_square += fact_count * fact_count
        passage_square += passage_count * passage_count
    fact_norm = math.sqrt(fact_square)
    passage_norm = math.sqrt(passage_square)
    if fact_norm == 0.0 or passage_norm == 0.0:
        return 0.0
    return dot / (fact_norm * passage_norm)


def cosine_score(fact, passages):
    """
    Maximum count-vector cosine of the fact against any passage.
    Tokens are the raw result of text.split(), with case, punctuation and
    tags preserved. Returns None when the fact has no tokens or no passages
    are supplied. Gold labels are not used.
    """
    fact_counts = Counter(fact.split())
    if len(fact_counts) == 0 or not passages:
        return None
    best_score = 0.0
    for passage in passages:
        passage_counts = Counter(passage["text"].split())
        score = cosine_similarity(fact_counts, passage_counts)
        if score > best_score:
            best_score = score
    return best_score


def predict_cosine(fact, passages, threshold):
    """Return S or NS. An empty fact or no passages is always NS."""
    score = cosine_score(fact, passages)
    if score is None:
        return "NS"
    if score >= threshold:
        return "S"
    return "NS"


def evaluation_gold(label):
    """Map IR to NS for evaluation only. S and NS stay unchanged."""
    if label == "IR":
        return "NS"
    return label


def evaluate(examples, golds, predict):
    """Score one method on the shared examples. predict does not receive gold labels."""
    rows = []
    for threshold in [i / 20 for i in range(21)]:
        predictions = [
            predict(example.fact, example.passages, threshold)
            for example in examples
        ]
        matrix = confusion_matrix(golds, predictions)
        correct = matrix[0][0] + matrix[1][1]
        total = sum(sum(row) for row in matrix)
        rows.append({
            "threshold": threshold,
            "correct": correct,
            "total": total,
            "accuracy_percent": (100.0 * correct / total) if total else 0.0,
            "matrix": matrix,
        })
    best_index = max(range(len(rows)), key=lambda index: (rows[index]["correct"], index))
    return rows, rows[best_index]


def confusion_matrix(golds, predictions):
    labels = ["S", "NS"]
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0, 0], [0, 0]]
    for gold, prediction in zip(golds, predictions):
        matrix[index[gold]][index[prediction]] += 1
    return matrix


def print_results(rows, best):
    print("Before preprocessing: split-only Word Recall")
    print("Development-set-tuned baseline accuracy")
    print("Threshold chosen and scored on the supplied development facts.")
    print()
    print_threshold_table(rows, best)


def print_cosine_results(rows, best):
    print()
    print("Before preprocessing: split-only Count Vectors + Cosine Similarity")
    print("Development-set-tuned baseline accuracy")
    print("Threshold chosen and scored on the supplied development facts.")
    print()
    print_threshold_table(rows, best)


def print_threshold_table(rows, best):
    print(f"{'Threshold':>9} | {'Correct':>7} | {'Total':>5} | {'Accuracy (%)':>12}")
    print("-" * 44)
    for row in rows:
        print(
            f"{row['threshold']:9.2f} | "
            f"{row['correct']:7d} | "
            f"{row['total']:5d} | "
            f"{row['accuracy_percent']:12.2f}"
        )
    print()
    print(f"Best threshold: {best['threshold']:.2f}")
    print(
        "Development-set-tuned baseline accuracy: "
        f"{best['correct']}/{best['total']} = {best['accuracy_percent']:.2f}%"
    )
    matrix = best["matrix"]
    print()
    print("Confusion matrix")
    print("Rows are gold labels. Columns are predictions. Order: S, NS.")
    print("Gold IR labels were mapped to NS for evaluation only.")
    print(f"{'':8} {'Pred S':>8} {'Pred NS':>8}")
    print(f"{'Gold S':8} {matrix[0][0]:8d} {matrix[0][1]:8d}")
    print(f"{'Gold NS':8} {matrix[1][0]:8d} {matrix[1][1]:8d}")


def print_comparison(word_best, cosine_best):
    print()
    print("Development-set-tuned, split-only baselines")
    print(
        f"{'Method':<28} | {'Best threshold':>15} | "
        f"{'Correct / Total':>16} | {'Accuracy (%)':>12}"
    )
    print("-" * 80)
    for name, best in (
        ("Word Recall", word_best),
        ("Count Vectors + Cosine", cosine_best),
    ):
        ratio = f"{best['correct']}/{best['total']}"
        print(
            f"{name:<28} | "
            f"{best['threshold']:15.2f} | "
            f"{ratio:>16} | "
            f"{best['accuracy_percent']:12.2f}"
        )


def main():
    root = Path(__file__).resolve().parent
    labels_path = root / "data" / "dev_labeled_ChatGPT.jsonl"
    passages_path = root / "data" / "passages_bm25_ChatGPT_humfacts.jsonl"

    fact_to_passages = read_passages(str(passages_path))
    examples = read_fact_examples(str(labels_path), fact_to_passages)
    golds = [evaluation_gold(example.label) for example in examples]

    word_rows, word_best = evaluate(examples, golds, predict_word_recall)
    cosine_rows, cosine_best = evaluate(examples, golds, predict_cosine)
    print_results(word_rows, word_best)
    print_cosine_results(cosine_rows, cosine_best)
    print_comparison(word_best, cosine_best)


if __name__ == "__main__":
    main()
