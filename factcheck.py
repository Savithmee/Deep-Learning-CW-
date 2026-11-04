# factcheck.py

import re
import html
import unicodedata
import torch
from typing import List, Dict, Optional
import numpy as np
import spacy
import gc


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


# =====================================================================================================
# PASSAGE PREPARATION AND PRUNING  (Person D)
# =====================================================================================================

STOPWORDS = set("""a about above after again against all am an and any are as at be because been before being
below between both but by can did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of
off on once only or other our ours ourselves out over own same she should so some such than that the their theirs
them themselves then there these they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves also""".split())

# Final settings chosen by Person D (update from D's BEST_CFG)
DEFAULT_CFG = dict(
    splitter="regex",        # tags | naive | regex | nltk | spacy
    min_words=3,
    max_words=100,
    window=2,                # 1 = single sentences, 2 = two-sentence windows (final choice)
    stride=1,
    title="none",            # none | all | pronoun
)

# ---------------- cleaning
_TAG = re.compile(r"<[^>]+>")
_CITE = re.compile(r"\[\d+\]|\[citation needed\]", re.I)


def clean_text(text: str, title: Optional[str] = None) -> str:
    t = html.unescape(text)
    t = unicodedata.normalize("NFKC", t)
    t = _CITE.sub(" ", t)
    t = _TAG.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if title and t.startswith(title + " " + title):
        t = t[len(title) + 1:]
    return t


# ---------------- sentence splitting
_REGEX_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")
_SPACY = None
_NLTK_OK = None


def _split_text(segments: List[str], splitter: str) -> List[str]:
    global _SPACY, _NLTK_OK
    if splitter == "tags":
        return segments
    out = []
    if splitter == "naive":
        for s in segments:
            parts = s.split(". ")
            out += [p if p.endswith(".") or i == len(parts) - 1 else p + "." for i, p in enumerate(parts)]
    elif splitter == "regex":
        for s in segments:
            out += _REGEX_SPLIT.split(s)
    elif splitter == "nltk":
        import nltk
        from nltk.tokenize import sent_tokenize
        if _NLTK_OK is None:
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
            _NLTK_OK = True
        for s in segments:
            out += sent_tokenize(s)
    elif splitter == "spacy":
        if _SPACY is None:
            _SPACY = spacy.blank("en")
            _SPACY.add_pipe("sentencizer")
        for doc in _SPACY.pipe(segments):
            out += [x.text for x in doc.sents]
    else:
        raise ValueError("unknown splitter " + splitter)
    return out


def _chunk(text: str, max_words: Optional[int]) -> List[str]:
    words = text.split()
    if not max_words or len(words) <= max_words:
        return [text]
    pieces, cur = [], []
    for part in re.split(r"(?<=;)\s+", text):
        pw = part.split()
        if cur and len(cur) + len(pw) > max_words:
            pieces.append(" ".join(cur))
            cur = []
        cur += pw
    if cur:
        pieces.append(" ".join(cur))
    out = []
    for p in pieces:
        w = p.split()
        out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)]
    return out


def sentence_units(passages: List[dict], splitter="regex", min_words=3, max_words=100) -> List[Dict]:
    units, seen = [], set()
    for pid, p in enumerate(passages):
        title = p.get("title", "")
        segs = [clean_text(s, title) for s in re.split(r"</?s>", p["text"])]
        segs = [s for s in segs if s]
        sid = 0
        for sent in _split_text(segs, splitter):
            for piece in _chunk(sent.strip(), max_words):
                if len(piece.split()) < min_words:
                    continue
                key = piece.lower()
                if key in seen:
                    continue
                seen.add(key)
                units.append(dict(title=title, pid=pid, sid=sid, text=piece))
                sid += 1
    return units


def _windows(units: List[Dict], window: int, stride: int) -> List[Dict]:
    if window <= 1:
        return [dict(title=u["title"], text=u["text"]) for u in units]
    out, seen = [], set()
    pids = sorted({u["pid"] for u in units})
    for pid in pids:
        g = [u for u in units if u["pid"] == pid]
        n = len(g)
        starts = [0] if window >= n else list(range(0, n - window + 1, max(1, stride)))
        if window < n and starts[-1] != n - window:
            starts.append(n - window)
        for s in starts:
            text = " ".join(u["text"] for u in g[s:s + window])
            if text.lower() not in seen:
                seen.add(text.lower())
                out.append(dict(title=g[0]["title"], text=text))
    return out


def candidates(passages: List[dict], cfg: Optional[dict] = None) -> List[Dict]:
    c = {**DEFAULT_CFG, **(cfg or {})}
    units = sentence_units(passages, c["splitter"], c["min_words"], c["max_words"])
    return _windows(units, c["window"], c["stride"])


# ---------------- overlap + pruning
def content_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def overlap_score(fact_tokens: set, text: str) -> float:
    if not fact_tokens:
        return 1.0
    return len(fact_tokens & content_tokens(text)) / len(fact_tokens)


def select(scores: List[float], min_overlap: float, min_keep: int = 1, max_keep: Optional[int] = None) -> List[int]:
    if not scores:
        return []
    keep = [i for i, s in enumerate(scores) if s >= min_overlap]
    if max_keep is not None and len(keep) > max_keep:
        keep = sorted(sorted(keep, key=lambda i: -scores[i])[:max_keep])
    if not keep:
        keep = sorted(sorted(range(len(scores)), key=lambda i: -scores[i])[:max(1, min_keep)])
    return keep


def prune(fact: str, sentences: List[str], min_overlap: float = 0.1,
          min_keep: int = 1, max_keep: Optional[int] = None) -> List[str]:
    ft = content_tokens(fact)
    scores = [overlap_score(ft, s) for s in sentences]
    return [sentences[i] for i in select(scores, min_overlap, min_keep, max_keep)]


# ---------------- titles
_PRON = re.compile(r"^(He|She|They|His|Her|Their|It|Its)\b")


def _with_title(title: str, text: str, mode: str) -> str:
    if mode == "all" or (mode == "pronoun" and _PRON.match(text)):
        return f"{title}: {text}"
    return text


def split_passages(passages: List[dict], **cfg) -> List[str]:
    c = {**DEFAULT_CFG, **cfg}
    return [_with_title(x["title"], x["text"], c["title"]) for x in candidates(passages, c)]


# =====================================================================================================
# ENTAILMENT MODEL AND DECISION STRATEGY  (Person C)
# =====================================================================================================

class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda
        self.max_length = max_length
        self.model.eval()
        # Read the label order from the model config instead of assuming it.
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        self.ent_idx = next(i for i, l in id2label.items() if l.startswith("entail"))
        self.neu_idx = next(i for i, l in id2label.items() if l.startswith("neutral"))
        self.con_idx = next(i for i, l in id2label.items() if l.startswith("contra"))

    def check_entailment(self, premise, hypothesis: str):
        """
        :param premise: one premise string, or a list of premise strings (batched)
        :param hypothesis: the fact
        :return: np.ndarray of shape (n, 3); columns are [entailment, neutral, contradiction] probabilities,
        always in this fixed order regardless of the model's internal label order
        """
        premises = [premise] if isinstance(premise, str) else list(premise)
        with torch.no_grad():
            # Tokenize the premises and hypothesis
            inputs = self.tokenizer(premises, [hypothesis] * len(premises), return_tensors='pt',
                                    truncation=True, max_length=self.max_length, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs = probs[:, [self.ent_idx, self.neu_idx, self.con_idx]]

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()

        return probs


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")   # Part 1 (word overlap): paste the teammate's implementation here


class EntailmentFactChecker(FactChecker):
    # ---- settings chosen from the experiments (final notebook run) ----
    STRATEGY = "ent_minus_con"   # "argmax" | "p_ent" | "ent_minus_con"
    AGG = "max"             # "max" | "top2" | "mean"
    THRESHOLD = 0.3
    MIN_OVERLAP = 0.2
    BATCH = 32

    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        sents = split_passages(passages)                   # Person D: cleaning and sentence splitting
        sents = prune(fact, sents, self.MIN_OVERLAP)       # Person D: word-overlap pruning
        if not sents:
            return "NS"
        probs = np.concatenate([self.ent_model.check_entailment(sents[i:i + self.BATCH], fact)
                                for i in range(0, len(sents), self.BATCH)])
        if self.STRATEGY == "argmax":
            # discrete: supported if ANY sentence is classified as entailment
            return "S" if (probs.argmax(axis=1) == 0).any() else "NS"
        v = probs[:, 0] if self.STRATEGY == "p_ent" else probs[:, 0] - probs[:, 2]
        if self.AGG == "max":
            score = v.max()
        elif self.AGG == "top2":
            score = np.sort(v)[-2:].mean()
        else:
            score = v.mean()
        return "S" if score > self.THRESHOLD else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
