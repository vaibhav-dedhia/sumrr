# Tyler Truong 3221610
# two
# three
# Professor Fei Liu
# NLP

import collections
import itertools
import xml.etree.ElementTree
import nltk
import numpy as np
import os
import re

# Note xml parser crashes on html entities (&amp; &lt; &gt; &eq;)
# All entities are replaced with '' emtpy string
strip_html_entities = re.compile(r'&[A-Za-z]+;')

# basic list of stopwords
stopwords = nltk.corpus.stopwords.words('english')

# returns a list of sentences and their corresponding tokens
def parse(filepath):
    with open(filepath) as f:
	    data = f.read()
    data = strip_html_entities.sub('', data)
    text = xml.etree.ElementTree.fromstring(data).find('TEXT').text

    tokens = []
    sentences = nltk.tokenize.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.tokenize.word_tokenize(sentence)
        words = [word.lower() for word in words]
        words = [word for word in words if word not in stopwords and word.isalnum()]
        tokens.append(words)
    return sentences, tokens

# performs summarization on each case
def summarize(casepath):
    inc = itertools.count()
    vocab = collections.defaultdict(lambda: inc.next())
    data = []
    text = []
    vec = []
    for _, _, files in os.walk(casepath):
        # ignore annoying apple files
        mask = ['._', '.DS_Store']
        valid_files = (file for file in files if not any(s in file for s in mask))
        for file in valid_files:
            filepath = os.path.join(casepath, file)
            a, b = parse(filepath)
            text += a
            data += b

    # build matrix of (sentence, norm vector)
    token_iter = (token for sentence in data for token in sentence)
    for token in token_iter:
        vocab[token]
    mat = np.zeros((len(data), len(vocab)))
    for i in range(len(data)):
        for token in data[i]:
            mat[i][vocab[token]] += 1
    mat = mat / (1 + mat.sum(axis=1)[:, np.newaxis])

    # MMR
    # Centroid is average of all sentences, used as Query parameter
    centroid = mat.mean(axis=0)
    top = (mat * centroid).sum(axis=1)
    bottom = ((centroid**2).sum()**0.5) * ((mat**2).sum(axis=1)**0.5)
    sim = top / (bottom + 1)
    print text[sim.argmax()]
    # Initial. Grap closest sentence to Centroid
    # Iterations: Grab max{Y * sim(S_unselected, Centroid) -
    #               (1 - Y) max[sim(S_unselected, S_selected)]}

# main iteration through all of the directories.
for root, cases, _ in os.walk('Documents'):
    for case in cases:
        casepath = os.path.join(root, case)
        summarize(casepath)
