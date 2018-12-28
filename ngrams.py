#!/usr/bin/env python3

import string
import sys
import nltk

VOCAB = set(string.punctuation) | {
    'a', 'akesi', 'ala', 'alasa', 'ale', 'ali', 'anpa', 'ante', 'anu', 'apeja', 'awen',
    'e', 'en', 'esun',
    'ijo', 'ike', 'ilo', 'insa',
    'jaki', 'jan', 'jelo', 'jo',
    'kala', 'kalama', 'kama', 'kasi', 'ken', 'kepeken', 'kijetesantakalu', 'kili', 'kin', 'kipisi',
    'kiwen', 'ko', 'kon', 'kule', 'kulupu', 'kute',
    'la', 'lape', 'laso', 'lawa', 'leko', 'len', 'lete', 'li', 'lili', 'linja', 'lipu', 'loje',
    'lon', 'luka', 'lukin', 'lupa',
    'ma', 'majuna', 'mama', 'mani', 'meli', 'mi', 'mije', 'moku', 'moli', 'monsi', 'monsuta', 'mu',
    'mun', 'musi', 'mute',
    'namako', 'nanpa', 'nasa', 'nasin', 'nena', 'ni', 'nimi', 'noka',
    'o', 'oko', 'olin', 'ona', 'open',
    'pakala', 'pali', 'palisa', 'pan', 'pana', 'pi', 'pilin', 'pimeja', 'pini', 'pipi', 'po',
    'poka', 'poki', 'pona', 'powe', 'pu',
    'sama', 'seli', 'selo', 'seme', 'sewi', 'sijelo', 'sike', 'sin', 'sina', 'sinpin', 'sitelen',
    'sona', 'soweli', 'suli', 'suno', 'supa', 'suwi',
    'tan', 'taso', 'tawa', 'telo', 'tenpo', 'toki', 'tomo', 'tu', 'tuli',
    'unpa', 'uta', 'utala',
    'walo', 'wan', 'waso', 'wawa', 'weka', 'wile'}

CORPUS = nltk.corpus.PlaintextCorpusReader('./Corpus/', r'.*\.txt')

def normalise(sent):
    sent = [word.lower() if word.lower() in VOCAB else 'X' for word in sent]
    if sent.count('X') > len(sent)/2 or 'XXXX' in ''.join(sent):
        return []
    return sent

def main():
    n = 1
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    fdist = nltk.FreqDist(' '.join(ngram)
                          for fid in CORPUS.fileids()
                          for sent in CORPUS.sents(fid)
                          for ngram in nltk.ngrams(normalise(sent), n))
    for ngram, freq in fdist.most_common():
        if all(word in 'X' + string.punctuation for word in ngram.split()):
            continue
        if freq < 2 or (len(sys.argv) > 2 and freq < int(sys.argv[2])):
            break
        print(freq, ngram)

if __name__ == '__main__':
    main()
