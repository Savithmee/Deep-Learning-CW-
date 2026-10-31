"""
Before preprocessing: four split-only baselines.

Word Recall, Jaccard, passage-local TF-IDF + Cosine, and ROUGE-1 F1 are
scored in this file. Tokenization is text.split() only. The score functions
accept token lists and do not call WordRecallThresholdFactChecker, so later
preprocessing inside factcheck.py does not change these results.

Each printed accuracy is a development-set-tuned baseline accuracy: the
threshold is chosen on the same labeled development facts that are scored.
"""

import math
from collections import Counter
from pathlib import Path

from factchecking_main import read_fact_examples, read_passages


THRESHOLDS = [i / 20 for i in range(21)]

TFIDF_SCOPE = [
    "TF-IDF fitting scope: passage-local.",
    "For each fact, vocabulary and IDF are fit only on that fact's supplied passages.",
    "That same vocabulary and IDF are used for the fact and all of those passages.",
    "A separate vectorizer is not fit for each fact-passage pair.",
    "IDF is not fit on the full dataset.",
    "TF is the raw token count.",
    "IDF is natural log((1 + N) / (1 + df)) + 1.",
    "N is the number of supplied passages.",
    "df is the number of those passages that contain the token.",
    "Fact tokens outside the fitted vocabulary are ignored.",
    "Cosine uses the complete vocabulary, including zero weights, in both norms.",
    "Similarity with a zero vector is 0.",
]


def tokenize(text):
    """Split-only tokens. Order, case, punctuation, tags and repetitions stay."""
    return text.split()


def word_recall(fact_tokens, passage_tokens):
    """|unique fact tokens intersect unique passage tokens| / |unique fact tokens|."""
    fact_types = set(fact_tokens)
    if len(fact_types) == 0:
        return 0.0
    passage_types = set(passage_tokens)
    return len(fact_types & passage_types) / len(fact_types)


def jaccard(fact_tokens, passage_tokens):
    """Intersection size / union size of the two token sets."""
    fact_types = set(fact_tokens)
    passage_types = set(passage_tokens)
    union = fact_types | passage_types
    if len(union) == 0:
        return 0.0
    return len(fact_types & passage_types) / len(union)


def passage_idf(passage_token_lists):
    """Fit one vocabulary and smooth IDF on this fact's passages."""
    document_frequency = Counter()
    for passage_tokens in passage_token_lists:
        document_frequency.update(set(passage_tokens))
    n_passages = len(passage_token_lists)
    idf = {}
    for token, df in document_frequency.items():
        idf[token] = math.log((1 + n_passages) / (1 + df)) + 1
    return idf


def tfidf_weights(tokens, idf):
    """Raw-count TF times the shared IDF. Dimensions are the fitted vocabulary."""
    counts = Counter(tokens)
    return {token: counts[token] * weight for token, weight in idf.items()}


def cosine(left_weights, right_weights):
    """Cosine over every shared dimension. A zero vector has similarity 0."""
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for token, left_weight in left_weights.items():
        right_weight = right_weights[token]
        dot += left_weight * right_weight
        left_square += left_weight * left_weight
        right_square += right_weight * right_weight
    left_norm = math.sqrt(left_square)
    right_norm = math.sqrt(right_square)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def tfidf_cosine(fact_tokens, passage_tokens, idf):
    """Cosine of TF-IDF vectors. Fact tokens absent from idf are ignored."""
    return cosine(tfidf_weights(fact_tokens, idf), tfidf_weights(passage_tokens, idf))


def rouge1_f1(fact_tokens, passage_tokens):
    """
    ROUGE-1 F1 from clipped unigram counts.
    Precision divides by the number of passage tokens, recall by fact tokens.
    A zero denominator or a zero precision-recall sum gives F1 0.
    """
    fact_count = len(fact_tokens)
    passage_count = len(passage_tokens)
    if fact_count == 0 or passage_count == 0:
        return 0.0
    fact_counts = Counter(fact_tokens)
    passage_counts = Counter(passage_tokens)
    overlap = 0
    for token in set(fact_counts) | set(passage_counts):
        overlap += min(fact_counts[token], passage_counts[token])
    precision = overlap / passage_count
    recall = overlap / fact_count
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def maximum_passage_score(fact_tokens, passage_token_lists, score_passage):
    """Highest passage score. None means the example must predict NS."""
    if len(fact_tokens) == 0 or not passage_token_lists:
        return None
    best_score = 0.0
    for passage_tokens in passage_token_lists:
        score = score_passage(fact_tokens, passage_tokens)
        if score > best_score:
            best_score = score
    return best_score


def maximum_word_recall(fact_tokens, passage_token_lists):
    return maximum_passage_score(fact_tokens, passage_token_lists, word_recall)


def maximum_jaccard(fact_tokens, passage_token_lists):
    return maximum_passage_score(fact_tokens, passage_token_lists, jaccard)


def maximum_tfidf_cosine(fact_tokens, passage_token_lists):
    """Fit IDF once on these passages, then take the maximum passage cosine."""
    if len(fact_tokens) == 0 or not passage_token_lists:
        return None
    idf = passage_idf(passage_token_lists)
    best_score = 0.0
    for passage_tokens in passage_token_lists:
        score = tfidf_cosine(fact_tokens, passage_tokens, idf)
        if score > best_score:
            best_score = score
    return best_score


def maximum_rouge1_f1(fact_tokens, passage_token_lists):
    return maximum_passage_score(fact_tokens, passage_token_lists, rouge1_f1)


def evaluation_gold(label):
    """Map IR to NS for evaluation only. S and NS stay unchanged."""
    if label == "IR":
        return "NS"
    return label


def predict_from_score(score, threshold):
    """Apply a threshold to a stored full-precision score. None is always NS."""
    if score is None:
        return "NS"
    if score >= threshold:
        return "S"
    return "NS"


def confusion_matrix(golds, predictions):
    labels = ["S", "NS"]
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0, 0], [0, 0]]
    for gold, prediction in zip(golds, predictions):
        matrix[index[gold]][index[prediction]] += 1
    return matrix


def evaluate_scores(scores, golds):
    """Reuse stored scores across the 21 thresholds. Gold is not inside the scores."""
    rows = []
    for threshold in THRESHOLDS:
        predictions = [predict_from_score(score, threshold) for score in scores]
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


def print_method(name, rows, best, notes=None):
    print()
    print("Before preprocessing: split-only " + name)
    print("Development-set-tuned baseline accuracy")
    print("Threshold chosen and scored on the supplied development facts.")
    if notes:
        for line in notes:
            print(line)
    print()
    print_threshold_table(rows, best)


def print_comparison(results):
    print()
    print("Development-set-tuned, split-only baselines")
    print(
        f"{'Method':<30} | {'Best threshold':>15} | "
        f"{'Correct / Total':>16} | {'Accuracy (%)':>12}"
    )
    print("-" * 82)
    for name, best in results:
        ratio = f"{best['correct']}/{best['total']}"
        print(
            f"{name:<30} | "
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
    prepared = []
    golds = []
    for example in examples:
        prepared.append((
            tokenize(example.fact),
            [tokenize(passage["text"]) for passage in example.passages],
        ))
        golds.append(evaluation_gold(example.label))

    methods = [
        ("Word Recall", maximum_word_recall, None),
        ("Jaccard Similarity", maximum_jaccard, None),
        ("TF-IDF + Cosine Similarity", maximum_tfidf_cosine, TFIDF_SCOPE),
        ("ROUGE-1 F1", maximum_rouge1_f1, None),
    ]
    comparison = []
    for name, score_example, notes in methods:
        scores = [
            score_example(fact_tokens, passage_token_lists)
            for fact_tokens, passage_token_lists in prepared
        ]
        rows, best = evaluate_scores(scores, golds)
        print_method(name, rows, best, notes)
        comparison.append((name, best))
    print_comparison(comparison)


if __name__ == "__main__":
    main()
