#!/usr/bin/env python3

import nltk
import string
import sys
import tp

def main():
    n = 1
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    fdist = [None] * (n+1)
    for k in range(n, 0, -1):
        print(k, file=sys.stderr, end=' ' if k > 1 else '\n', flush=True)
        fdist[k] = tp.ngram_dist(k)
    for k in range(n, 0, -1):
        print('#', str(k) + '-gram frequencies')
        for ngram, freq in fdist[k].most_common():
            if not tp.check_ngram(ngram):
                continue
            if freq < 2 or (len(sys.argv) > 2 and freq < int(sys.argv[2])):
                break
            print(freq, ngram)
            for j in range(1, k):
                for sub in nltk.ngrams(ngram.split(), j):
                    fdist[j][' '.join(sub)] -= freq

if __name__ == '__main__':
    main()
