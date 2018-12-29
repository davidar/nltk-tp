#!/usr/bin/env python3

import string
import sys
import tp

from nltk.collocations import *
from nltk.metrics.association import *

def main():
    n = 2
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if n == 2:
        metric = BigramAssocMeasures()
        finder = BigramCollocationFinder
    elif n == 3:
        metric = TrigramAssocMeasures()
        finder = TrigramCollocationFinder
    elif n == 4:
        metric = QuadgramAssocMeasures()
        finder = QuadgramCollocationFinder
    finder = finder.from_words(word for sent in tp.sents() for word in sent)
    finder.apply_freq_filter(50)
    finder.apply_word_filter(lambda w: w in string.punctuation)
    scored = finder.score_ngrams(metric.student_t)
    for ngram, score in scored:
        ngram = ' '.join(ngram)
        if not tp.check_ngram(ngram):
            continue
        if score < 1:
            break
        print(int(score), ngram)

if __name__ == '__main__':
    main()
