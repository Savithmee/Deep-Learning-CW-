# factcheck.py

import re
import torch
from typing import List, Optional
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


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


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
    _NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", 
    "twenty", "thirty", "forty", "fifty", "sixty", "hundred", "thousand", "million", "billion"} 
    # Number words that spaCy lists as stop words. They are removed from the stop list by default
    _stop_words, _porter, _nlp = None, None, None 
    
    def __init__(self, threshold: float = 0.633):
        """
        :param threshold: predict "S" when the maximum passage word-recall is
            greater than or equal to this value. 0.633 is the cutoff tuned for
            stop-word removal plus Porter stemming on the development facts.
        """
        self.threshold = threshold

    def preprocess(self, text):
        """Split into letter/digit words, drop stop words, then Porter-stem."""
        toks = re.findall(r"[A-Za-z0-9]+", text)
        if WordRecallThresholdFactChecker._stop_words is None:
            from spacy.lang.en.stop_words import STOP_WORDS
            WordRecallThresholdFactChecker._stop_words = set(STOP_WORDS) - WordRecallThresholdFactChecker._NUMBER_WORDS
        toks = [t for t in toks if t.lower() not in WordRecallThresholdFactChecker._stop_words]
        if WordRecallThresholdFactChecker._porter is None:
            from nltk.stem import PorterStemmer
            WordRecallThresholdFactChecker._porter = PorterStemmer()
        return [WordRecallThresholdFactChecker._porter.stem(t, to_lowercase=False) for t in toks]

    def overlap_score(self, fact: str, passages: List[dict]) -> Optional[float]:
        """
        Maximum word recall of the fact against any passage.
        Tokens come from preprocess: letter/digit words, stop words removed, Porter stemmed.
        Returns None when the fact has no tokens or no passages are supplied.
        Gold labels are not an input to this method.
        """
        fact_tokens = set(self.preprocess(fact))
        if len(fact_tokens) == 0 or not passages:
            return None
        best_score = 0.0
        for passage in passages:
            passage_tokens = set(self.preprocess(passage["text"]))
            score = len(fact_tokens & passage_tokens) / len(fact_tokens)
            if score > best_score:
                best_score = score
        return best_score

    def predict(self, fact: str, passages: List[dict]) -> str:
        score = self.overlap_score(fact, passages)
        if score is None:
            return "NS"
        if score >= self.threshold:
            return "S"
        return "NS"

    


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


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
