# understanding overall concept of the assignment: Part 01 Word Overlap

# given a statement eg: "Albert Einstein was born in Germany"
# given documentation of biographies of people [similar to a dictionary]
# question: does the documentation give sufficient support for the statement?
# task: implementing a system that will be capable of answering the above question effectively 

# Preprocessing: Firstly we should investigate the data to make informed decisions on the preprocessing steps

# Investigation:

# Connect each statement to its relevant documents
# Extract the number of statements 
# Investigate which words of the statements appear in the documents 
# Investigate whether there are strange symbols 
# Investigate whether words in the statements are missing in the documents
# Understand whether common words are useful for fact-checking 
# Understand the overlap between the statements and the documents

import json, re, sys, unicodedata
from collections import Counter 

LABELS = sys.argv[1] if len(sys.argv) > 1 else "data/dev_labeled_ChatGPT.jsonl"
PASSAGES = sys.argv[2] if len(sys.argv) > 2 else "data/passages_bm25_ChatGPT_humfacts.jsonl"

# -- Step 1: connecting each statement to its document and identifying the class balance

fact2pass = {}
for line in open(PASSAGES, encoding="utf-8"):
    d = json.loads(line) # converting each line to a python dictionary 
    fact2pass[d["sent"]] = d["passages"] # connecting each fact to its passages

exs = [] # creating a list for the examples
for line in open(LABELS, encoding="utf-8"):
    d = json.loads(line) # converting each line to a python dictionary 
    for sent in d["annotations"] or []:
        for f in sent["human-atomic-facts"] or []:
            lab = "NS" if f["label"] == "IR" else f["label"] # changing the label 'IR' to 'NS'
            exs.append((f["text"], lab, fact2pass[f["text"]]))
print("examples:", len(exs), Counter(l for _, l, _ in exs)) # prints dataset size and the class balance

# -- Step 2: investigating the passage characters

pchars = Counter(
    c 
    for _, _, ps in exs 
    for p in ps 
    for c in p["text"] 
    if not c.isalnum() and c != " " # keeping the characters that are not letters, numbers or a normal space
    # collects alphanumeric characters and punctuation 
    )
print("passage symbols:", pchars.most_common(12)) # printing the 12 most common passage symbols 

# same above process for the facts
fchars = Counter(
    c
    for f, _, _ in exs
    for c in f
    if not c.isalnum() and c != " "
)
print("fact symbols   :", fchars.most_common(12)) # printing the 12 most common fact symbols 

# searching for tag <s> [after going through dataset found many such tags]
print(
    "passages with <s>:", 
    sum("<s>" in p["text"] for _, _, ps in exs for p in ps)
) # printing the number of passages with the tag <s>

# checking whether there are non-ASCII characters 
nonascii_f = [
    f for f, _, _ in exs 
    if any(ord(c) > 127 for c in f)
    # checking whether characters have a unicode code point greater than 127 
] # gives a list of facts that contain atleast one non-ASCII character 
print("facts with non-ASCII:", len(nonascii_f), nonascii_f[:3])

# remove accents for comparison 
strip = lambda s: "".join(
    c for c in unicodedata.normalize("NFKD", s) 
    if not unicodedata.combining(c)
)
print(
    "facts whose accents change when stripped:", 
    sum(strip(f) != f for f, _, _ in exs)
) # printing the number of facts whose accents change when stripped

# checking for numbers, hyphens, apostrophes
# since these characters can change how words are split and compared

print(
    "facts with digits:", 
    sum(bool(re.search(r"\d", f)) for f, _, _ in exs)
) # printing the number of facts that contain atleast one digit

print(
    "facts with hyphen/apostrophe:", 
    sum(bool(re.search(r"[-']", f)) for f, _, _ in exs)
) # printing the number of facts that contain atleast one hyphen or apostrophe

# -- Step 3: tokenization 

def base(t):
    return re.findall(r"[a-z0-9]+", t.lower()) # finding all the words in the text
miss = Counter() # counting the words that are not found in the supported facts but cannot be found in their associated passages
for f, lab, ps in exs:
    if lab != "S": 
        continue # only Supported facts: their words SHOULD be found, so misses reveal cleaning gaps
    # collect the words from the passages
    pw = set(w for p in ps for w in base(p["text"])) # all passage words for this fact
    # finding the fact words that are missing from the passages
    miss.update(w for w in set(base(f)) if w not in pw)
print(
    "S-fact words missing from passages:", 
    miss.most_common(25)
) # printing the 25 most common words that are missing from passages

# -- Step 4: document frequency (most common words across passages)

uniq = {
    p["text"] 
    for _, _, ps in exs 
    for p in ps
} # collecting all the unique passages

df = Counter(
    w
    for t in uniq
    for w in set(base(t))
) # counting the frequency of each word across all passages
print("unique passages:", len(uniq)) # printing the number of unique passages
print(
    "top doc-freq words:",
    [(w, round(c/len(uniq), 2)) for w, c in df.most_common(15)]
) # printing the 15 most common words by document frequency

# -- Step 5: Stop-words 

SW = {
    "a", "an", "the", "is", "are", "was", "were", 
    "be", "been", "of", "in", "on", "at", "to",
    "for", "with", "by", "and", "or", "as", "from", 
    "that", "this", "it", "its", "he", "she", "his", 
    "her", "has", "have", "had"
}

# finding the share of stop-words in each fact
share = [
    sum(w in SW for w in base(f)) / max(1, len(base(f)))
    for f, _, _ in exs
] # gives a list of the share of stop-words in each fact

print(
    "mean stop-word share of a fact:",
    round(sum(share) / len(share), 2),
) # printing the average stop-word share

print(
    "facts EMPTY after stop-word removal:",
    sum(all(w in SW for w in base(f)) for f, _, _ in exs)
) # printing the number of facts that are empty after stop-word removal

# -- Step 6: Checking word coverage through recall score 

def recall(f, ps, drop=False):
    ft = [
        w for w in set(base(f)) 
        if not (drop and w in SW)
    ] # unique fact words 

    if not ft: 
        return 0.0 # guard against division by zero

    return max(
        sum(w in set(base(p["text"])) for w in ft) / len(ft)
        for p in ps
    ) # best passage: fraction of fact words found

# -- Step 7: Comparing 'S' and 'NS' facts 

for drop in (False, True):
    for lab in ("S", "NS"):
        v = [
            recall(f, ps, drop) 
            for f, l, ps in exs 
            if l == lab
        ]
        print(
            f"stop words {'dropped' if drop else 'kept   '} | "
            f"{lab:2s} mean recall = {sum(v)/len(v):.3f}"
        ) # printing the avg recall for the supported and non-supported facts

# -- Step 8: Repeated passages (cache evidence)

slots = sum(len(ps) for _, _, ps in exs) # total passage slots across all facts
print(
    f"passage slots {slots} vs unique passages {len(uniq)} "
    f"-> reuse {slots/len(uniq):.1f}x"
) # printing the ratio of passage slots to unique passages

