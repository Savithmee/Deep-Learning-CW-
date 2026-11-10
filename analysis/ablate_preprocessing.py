# Run: python ablate_preprocessing.py [labels_path] [passages_path]
# Finding the best preprocessing for word overlap method 

# In this script we focus only on the preprocessing methodology 

# How accuracy is obtained when only preprocessing changes?
# 1. The scorer never changes: score = (unique fact tokens found in a passage) / (unique fact tokens),
#    best passage wins (max over passages).
# 2. A threshold turns score into a label: score >= threshold -> "S", otherwise "NS".
#    We try a threshold between every pair of neighbouring scores and keep the best one.
# 3. accuracy = (facts whose predicted label equals the gold label) / 221.

import json, re, sys, unicodedata
from collections import namedtuple
from typing import List
import numpy as np

MIN_GAIN = 0.01 # a particular step must add >=1 accuracy point 
K_FOLDS = 5

Example = namedtuple("Example", "fact label passages group")      
# group = which biography
# (helps in cross-validation where facts belonging to same biography do not get split across training & testing folds)

def load_examples(labels_path="data/dev_labeled_ChatGPT.jsonl", passages_path="data/passages_bm25_ChatGPT_humfacts.jsonl"):
    fact2pass = {}
    for line in open(passages_path, encoding="utf-8"):
        d = json.loads(line)
        fact2pass[d["sent"]] = d["passages"] # connecting each fact to its passages
    exs = []
    for line in open(labels_path, encoding="utf-8"):
        d = json.loads(line)
        for sent in d["annotations"] or []:
            for f in sent["human-atomic-facts"] or []:
                lab = "NS" if f["label"] == "IR" else f["label"] # changing the label 'IR' to 'NS'
                exs.append(Example(f["text"], lab, fact2pass[f["text"]], d["topic"])) # fact, label, passages, biography
    return exs

# Preprocessing Steps 
# tokenisation itself is always on

CANDIDATES = ["markup", "thousands", "accents", "possessive", "lower", "stop", "porter", "lemma"]
EXCLUSIVE = {"porter": "lemma", "lemma": "porter"} # stemming and lemmatisation do the same job (never both)

_NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", 
"twenty", "thirty", "forty", "fifty", "sixty", "hundred", "thousand", "million", "billion"} 
# Number words that spaCy lists as stop words. They are removed from the stop list by default
_stop_words, _porter, _nlp = None, None, None 

def preprocess(text,steps):
    global _stop_words, _porter, _nlp

    # Step 01: Remove markup tags.                 
    if "markup" in steps:
        text = re.sub(r"<[^>]*>", " ", text)
    
    # Step 02: Join thousands separators so "1,942" stays ONE token instead of "1" + "942".
    if "thousands" in steps:
        text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    
    # Step 03: Strip accents.
    if "accents" in steps:
        text = "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )
    
    # Step 04: Strip the possessive 's  ("world's" -> "world").
    if "possessive" in steps:
        text = re.sub(r"'s\b", "", text)

    # Step 05: Tokenise [always]
    if "lemma" in steps:
        if _nlp is None:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        raw = [t.lemma_ for t in _nlp(text)]
        toks = [w for t in raw for w in re.findall(r"[A-Za-z0-9]+", t)]
    else:
        toks = re.findall(r"[A-Za-z0-9]+", text)

    # Step 06: Lowercase.
    if "lower" in steps:
        toks = [t.lower() for t in toks]

    # Step 07: Remove stop words.
    if "stop" in steps:
        if _stop_words is None:
            from spacy.lang.en.stop_words import STOP_WORDS
            _stop_words = set(STOP_WORDS) - _NUMBER_WORDS
        toks = [t for t in toks if t.lower() not in _stop_words]
    
    # Step 08: Normalise.
    # Porter stemming 
    if "porter" in steps:
        if _porter is None:
            from nltk.stem import PorterStemmer
            _porter = PorterStemmer()
        toks = [_porter.stem(t, to_lowercase=("lower" in steps)) for t in toks]
    return toks

# Scoring method which is fixed plus cache to make lemmatisation fast
_TOKEN_CACHE = {}
_RESULT_CACHE = {}

def _token_set(text, steps):
    key = (text, steps)                                 
    if key not in _TOKEN_CACHE:
        _TOKEN_CACHE[key] = frozenset(preprocess(text,steps))
    return _TOKEN_CACHE[key]

def overlap_score(fact, passages, steps):
    ft = _token_set(fact, steps)
    if not ft:
        return 0.0                                    
    return max(
        (
            len(ft & _token_set(p["text"], steps)) / len(ft) 
            for p in passages
        ), 
        default=0.0
    )

# Threshold tuning, cross-validation

def tune_threshold(scores, gold):
    # finding the best threshold between every pair of neighbouring scores; which gives the highest accuracy
    s, g = np.asarray(scores, float), np.asarray(gold, bool)
    u = np.unique(s)
    cands = np.concatenate(
        ([u[0] - 1e-9],
         (u[:-1] + u[1:]) / 2,
         [u[-1] + 1e-9])
    )
    accs = np.array([((s >= t) == g).mean() for t in cands])
    best = accs.max()
    return float(best), float(np.median(cands[accs == best]))

# cross-validation [ created by biography]

def cv_accuracy(scores, gold, groups, k=K_FOLDS, seed=0):
    # whole biographies go to one fold, so facts from the same person never sit in both train and test
    # the threshold is tuned on the training folds only and then judged on the unseen fold
    s, g = np.asarray(scores, float), np.asarray(gold, bool)
    names = sorted(set(groups))
    np.random.RandomState(seed).shuffle(names)
    fold_of = {name: i % k for i, name in enumerate(names)}
    folds = np.array([fold_of[x] for x in groups])
    correct = 0
    for f in range(k):
        train, test = folds != f, folds == f
        _, thr = tune_threshold(s[train], g[train])      
        correct += int(((s[test] >= thr) == g[test]).sum())
    return correct / len(s)

# experiment

def main(labels_path=None, passages_path=None):
    kw = {k: v for k, v in (("labels_path", labels_path), ("passages_path", passages_path)) if v}
    exs = load_examples(**kw)
    gold = [e.label == "S" for e in exs]              
    groups = [e.group for e in exs]    
    # summary of dataset                          
    print("examples: %d | S: %d | NS: %d | always-S baseline: %.2f%%" % (len(exs), sum(gold), len(gold) - sum(gold), 100 * np.mean(gold)))

    candidates = list(CANDIDATES)
    try:
        preprocess("test", frozenset(["lemma"]))
    except Exception as err:
        print("spaCy lemmatisation skipped:", repr(err)[:80])
        candidates.remove("lemma")

    def evaluate(steps):
        steps = frozenset(steps)
        if steps not in _RESULT_CACHE:
            scores = [overlap_score(e.fact, e.passages, steps) for e in exs]
            acc, thr = tune_threshold(scores, gold)
            _RESULT_CACHE[steps] = dict(acc=acc, thr=thr, cv=cv_accuracy(scores, gold, groups), scores=scores)
        return _RESULT_CACHE[steps]

    def blocked(step, chosen):
        return step in chosen or EXCLUSIVE.get(step) in chosen

    # forward selection: start with nothing, add the single most helpful step each round. stop when the best remaining step no longer gains MIN_GAIN
    chosen = frozenset()
    best_cv = evaluate(chosen)["cv"]
    print("\nStart (tokenise only): tuned %.2f%% | CV %.2f%%" % (100 * evaluate(chosen)["acc"], 100 * best_cv))
    round_no = 0

    while True:
        round_no += 1
        trials = [(evaluate(chosen | {s})["cv"], s) for s in candidates if not blocked(s, chosen)]
        if not trials:
            break
        print("\nRound %d: CV accuracy when each step is added to %s" % (round_no, sorted(chosen) or "nothing"))
        for cv, s in sorted(trials, reverse=True):
            print("  + %-11s CV %.2f%%  (%+.2f points)" % (s, 100 * cv, 100 * (cv - best_cv)))
        cv, s = max(trials, key=lambda t: t[0])
        if cv - best_cv < MIN_GAIN:
            print("-> best gain is below %.0f point(s): STOP adding steps" % (100 * MIN_GAIN))
            break
        chosen, best_cv = chosen | {s}, cv
        print("-> ADD '%s'" % s)
 
    # Backward check: remove any step whose removal costs less than MIN_GAIN (kills accidental extras)
    for s in sorted(chosen):
        without = evaluate(chosen - {s})["cv"]
        if best_cv - without < MIN_GAIN:
            print("\nBACKWARD CHECK: '%s' is not needed (removing it costs %.2f points) -> REMOVED"
                  % (s, 100 * (best_cv - without)))
            chosen, best_cv = chosen - {s}, evaluate(chosen - {s})["cv"]
 
    final = evaluate(chosen)
 
    # Final Report which preprocessing steps to use 
    print("RECOMMENDED PREPROCESSING")
    print("  %-11s %-6s %s" % ("step", "DECISION", "evidence (change in CV accuracy if toggled)"))
    for s in candidates:
        if s in chosen:
            delta = evaluate(chosen - {s})["cv"] - final["cv"]
            print("  %-11s USE: removing it changes CV by %+.2f points" % (s, 100 * delta))
        elif EXCLUSIVE.get(s) in chosen:
            print("  %-11s SKIP: %s already used (never use both)" % (s, EXCLUSIVE[s]))
        else:
            delta = evaluate(chosen | {s})["cv"] - final["cv"]
            print("  %-11s SKIP: adding it changes CV by %+.2f points (not worth the extra step)" % (s, 100 * delta))
    print("  (tokenisation to letters/digits is always used)")
 
    gold_arr, pred = np.asarray(gold, bool), np.asarray(final["scores"]) >= final["thr"]
    tp, fn = int((gold_arr & pred).sum()), int((gold_arr & ~pred).sum())
    fp, tn = int((~gold_arr & pred).sum()), int((~gold_arr & ~pred).sum())
    print("\nFinal steps: %s" % (sorted(chosen) or "tokenise only"))
    print("Threshold: %.3f" % final["thr"])
    print("Tuned accuracy: %.2f%%  (%d/%d)  <- target is >= 75%%" % (100 * final["acc"], tp + tn, len(exs)))
    print("%d-fold CV acc : %.2f%%  <- honest estimate on unseen biographies" % (K_FOLDS, 100 * final["cv"]))
    print("Confusion matrix (rows gold S/NS, cols predicted S/NS):\n  %d\t%d\n  %d\t%d" % (tp, fn, fp, tn))
    print("\nUse in factcheck.py: preprocess(text, frozenset(%s)) with threshold %.3f" % (sorted(chosen), final["thr"]))
    print("Note: differences under %.0f point(s) (about %d facts) are noise; the simplest option is preferred."
          % (100 * MIN_GAIN, round(MIN_GAIN * len(exs))))
 
if __name__ == "__main__":
    main(*sys.argv[1:3])