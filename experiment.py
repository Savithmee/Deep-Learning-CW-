import math
import re
from collections import Counter
from pathlib import Path

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


VARIANT_A = "Porter stemming"
VARIANT_B = "POS lemmatization"
THRESHOLDS = [i / 20 for i in range(21)]


def load_lemmatizer():
    """Load en_core_web_sm once. This script does not install or download it."""
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError(
            "spaCy is required for POS-aware lemmatization and for the checker's stop words.\n"
            "Install it manually, then download the model manually:\n"
            "python -m pip install spacy==3.8.2\n"
            "python -m spacy download en_core_web_sm"
        ) from error
    try:
        from nltk.stem import PorterStemmer
        PorterStemmer()
    except ImportError as error:
        raise RuntimeError(
            "NLTK is required because WordRecallThresholdFactChecker.preprocess uses PorterStemmer.\n"
            "Install it manually:\n"
            "python -m pip install nltk==3.9.1"
        ) from error
    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "spaCy model en_core_web_sm is required for POS-aware lemmatization.\n"
            "Download it manually:\n"
            "python -m spacy download en_core_web_sm\n"
            "This experiment does not download models."
        )
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception as error:
        raise RuntimeError(
            "spaCy model en_core_web_sm could not be loaded.\n"
            "Download it manually:\n"
            "python -m spacy download en_core_web_sm\n"
            "Original error: " + repr(error)
        ) from error


def tokens_before_stem(text):
    """
    The statements in WordRecallThresholdFactChecker.preprocess before Porter stemming.
    preprocess() does not return this list, so Variant B calls these same statements.
    """
    from factcheck import WordRecallThresholdFactChecker
    toks = re.findall(r"[A-Za-z0-9]+", text)
    if WordRecallThresholdFactChecker._stop_words is None:
        from spacy.lang.en.stop_words import STOP_WORDS
        WordRecallThresholdFactChecker._stop_words = set(STOP_WORDS) - WordRecallThresholdFactChecker._NUMBER_WORDS
    return [t for t in toks if t.lower() not in WordRecallThresholdFactChecker._stop_words]


def lemmatize_tokens(tokens, nlp):
    """POS lemmas from a pretokenized Doc. Lemmas are not lowercased here."""
    if not tokens:
        return []
    from spacy.tokens import Doc
    doc = nlp(Doc(nlp.vocab, words=list(tokens)))
    observed = [token.text for token in doc]
    if len(doc) != len(tokens) or observed != list(tokens):
        raise RuntimeError(
            "POS-aware lemmatization changed token boundaries, count, or order. "
            + "Input tokens: "
            + repr(list(tokens))
            + ". spaCy Doc tokens: "
            + repr(observed)
            + ". No fallback was used."
        )
    return [token.lemma_ for token in doc]


def cached(function):
    """Cache token lists by the raw string. Gold labels are not used."""
    cache = {}

    def preprocess(text):
        if text not in cache:
            cache[text] = function(text)
        return cache[text]

    return preprocess


def print_preprocessing_description():
    print("Development-set-tuned results")
    print("Each combination selects its threshold on the supplied development facts.")
    print()
    print("Shared rules, matching WordRecallThresholdFactChecker.preprocess before stemming:")
    print("- Token regex: [A-Za-z0-9]+, applied to the original text.")
    print("- A token is removed when token.lower() is in spaCy English STOP_WORDS")
    print("  minus WordRecallThresholdFactChecker._NUMBER_WORDS.")
    print("- No HTML or markup cleaning.")
    print("- No whole-text lowercasing.")
    print("- No extra exceptions for not, no, or never.")
    print("- Token lists keep their order and repetitions.")
    print("- Lists are passed to the scoring functions, which choose their own representation.")
    print()
    print("Variant A calls WordRecallThresholdFactChecker.preprocess directly.")
    print("That method Porter-stems every remaining token with to_lowercase=False.")
    print("Variant B uses the same pre-stem tokens and does not stem them.")
    print("Variant B builds Doc(nlp.vocab, words=tokens), then calls nlp(doc).")
    print("The spaCy tokenizer is not run again.")
    print("This script does not lowercase Variant B input tokens or the returned lemmas.")
    print("spaCy may still change case inside lemma_.")
    print()
    print("Threshold grid: [i / 20 for i in range(21)].")
    print("The 21 thresholds are 0.00, 0.05, 0.10, ..., 1.00.")
    print("Each of the eight combinations evaluates all 21 thresholds on its cached scores.")
    print("A numeric score predicts S when score >= threshold; otherwise it predicts NS.")
    print("Empty preprocessed facts and examples without passages stay NS.")
    print("The selected threshold is the one with the highest development-set accuracy.")
    print("If several thresholds tie, the highest threshold is selected.")
    print("This grid does not select the finalized checker threshold.")


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


def outcome_at_threshold(scores, golds, threshold):
    """Score the stored values. None always predicts NS and is not numeric."""
    predictions = [predict_from_score(score, threshold) for score in scores]
    matrix = confusion_matrix(golds, predictions)
    correct = matrix[0][0] + matrix[1][1]
    total = sum(sum(row) for row in matrix)
    return {
        "threshold": threshold,
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "matrix": matrix,
    }


def evaluate_grid(scores, golds):
    """Reuse cached scores on the fixed 0.05 grid. Ties keep the highest threshold."""
    rows = [outcome_at_threshold(scores, golds, threshold) for threshold in THRESHOLDS]
    best = max(rows, key=lambda row: (row["correct"], row["threshold"]))
    return rows, best


def print_confusion(matrix):
    print("Confusion matrix")
    print("Rows are gold labels. Columns are predictions. Order: S, then NS.")
    print("Gold IR labels were mapped to NS for evaluation only.")
    print(f"{'':8} {'Pred S':>8} {'Pred NS':>8}")
    print(f"{'Gold S':8} {matrix[0][0]:8d} {matrix[0][1]:8d}")
    print(f"{'Gold NS':8} {matrix[1][0]:8d} {matrix[1][1]:8d}")
    print(
        "TP "
        + str(matrix[0][0])
        + ", FN "
        + str(matrix[0][1])
        + ", FP "
        + str(matrix[1][0])
        + ", TN "
        + str(matrix[1][1])
    )


def accuracy_percent(row):
    if row["total"] == 0:
        return 0.0
    return 100.0 * row["correct"] / row["total"]


def print_threshold_table(rows):
    print(f"{'Threshold':>9} | {'Correct':>7} | {'Total':>5} | {'Accuracy (%)':>12}")
    print("-" * 44)
    for row in rows:
        print(
            f"{row['threshold']:9.2f} | "
            f"{row['correct']:7d} | "
            f"{row['total']:5d} | "
            f"{accuracy_percent(row):12.2f}"
        )


def print_selection(variant, method, rows, best, notes=None):
    print()
    print("Preprocessing: " + variant)
    print("Scoring method: " + method)
    print("Development-set-tuned accuracy")
    if notes:
        for line in notes:
            print(line)
    print()
    print_threshold_table(rows)
    print()
    print(f"Best threshold: {best['threshold']:.2f}")
    print(
        "Development-set-tuned accuracy: "
        + f"{best['correct']}/{best['total']} = {accuracy_percent(best):.2f}%"
    )
    print_confusion(best["matrix"])


def print_comparison(results):
    print()
    print("Development-set-tuned accuracy")
    print(
        f"{'Preprocessing':<18} | {'Scoring method':<28} | {'Best threshold':>14} | "
        f"{'Correct/Total':>14} | {'Accuracy (%)':>12}"
    )
    print("-" * 98)
    for variant, method, best in results:
        ratio = f"{best['correct']}/{best['total']}"
        print(
            f"{variant:<18} | "
            f"{method:<28} | "
            f"{best['threshold']:14.2f} | "
            f"{ratio:>14} | "
            f"{accuracy_percent(best):12.2f}"
        )


def main():
    nlp = load_lemmatizer()
    from factcheck import WordRecallThresholdFactChecker
    from factchecking_main import read_fact_examples, read_passages

    checker = WordRecallThresholdFactChecker()
    root = Path(__file__).resolve().parent
    labels_path = root / "data" / "dev_labeled_ChatGPT.jsonl"
    passages_path = root / "data" / "passages_bm25_ChatGPT_humfacts.jsonl"
    fact_to_passages = read_passages(str(passages_path))
    examples = read_fact_examples(str(labels_path), fact_to_passages)
    raw_examples = [
        (example.fact, [passage["text"] for passage in example.passages])
        for example in examples
    ]
    golds = [evaluation_gold(example.label) for example in examples]
    methods = [
        ("Word Recall", maximum_word_recall, None),
        ("Jaccard Similarity", maximum_jaccard, None),
        ("TF-IDF + Cosine Similarity", maximum_tfidf_cosine, TFIDF_SCOPE),
        ("ROUGE-1 F1", maximum_rouge1_f1, None),
    ]
    variant_functions = [
        (VARIANT_A, cached(checker.preprocess)),
        (VARIANT_B, cached(lambda text: lemmatize_tokens(tokens_before_stem(text), nlp))),
    ]
    print_preprocessing_description()
    comparison = []
    for variant, preprocess in variant_functions:
        prepared = [
            (preprocess(fact), [preprocess(passage) for passage in passages])
            for fact, passages in raw_examples
        ]
        for method, score_example, notes in methods:
            scores = [
                score_example(fact_tokens, passage_token_lists)
                for fact_tokens, passage_token_lists in prepared
            ]
            rows, best = evaluate_grid(scores, golds)
            print_selection(variant, method, rows, best, notes)
            comparison.append((variant, method, best))
    print_comparison(comparison)


if __name__ == "__main__":
    main()
