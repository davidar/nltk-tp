import nltk
import string

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

def normalise_word(word):
    word = word.lower()
    if word == '"':
        word = "'"
    if word not in VOCAB:
        word = 'X'
    return word

def normalise(sent):
    sent = ['^'] + [normalise_word(word) for word in sent if word != '-']
    if sent.count('X') > len(sent)/2 or 'XXXX' in ''.join(sent):
        return []
    return sent

def sents():
    for fid in CORPUS.fileids():
        for sent in CORPUS.sents(fid):
            yield normalise(sent)

def ngram_dist(n):
    return nltk.FreqDist(' '.join(ngram) for sent in sents() for ngram in nltk.ngrams(sent, n))

def check_ngram(ngram):
    if all(word in 'X' + string.punctuation for word in ngram.split()):
        return False
    if ngram.startswith('X') or ngram.endswith('X') or 'X X' in ngram:
        return False
    return True
