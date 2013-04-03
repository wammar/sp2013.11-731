import sys
from marginalize import marginalize
import models

MAX_PHRASE_OPTIONS = 1000000

tm = models.TM('data/tm', MAX_PHRASE_OPTIONS)
lm = models.LM('data/lm')
spanishSentences = [tuple(line.strip().split()) for line in open('data/input', 'r')]
bestSentScores = [-9999 for i in range(0,55)]
bestSent = ['' for i in range(0, 55)]

for fileName in sys.argv[1:]:
    for sentNum, enSent in enumerate(open(fileName, 'r')):
        if len(enSent) < 1:
            continue
        score = marginalize(spanishSentences[sentNum], enSent, lm, tm)
        if score > bestSentScores[sentNum]:
            bestSentScores[sentNum] = score
            bestSent[sentNum] = enSent
            
for sent in bestSent:
    print sent.strip()
