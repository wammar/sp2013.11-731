import math
import logging
from collections import Counter

def smooth_bleu_stats(hypothesis, reference):
    yield len(hypothesis)
    yield len(reference)
    for n in xrange(1, 5):
        s_ngrams = Counter(tuple(hypothesis[i:i+n]) for i in xrange(len(hypothesis)+1-n))
        r_ngrams = Counter(tuple(reference[i:i+n]) for i in xrange(len(reference)+1-n))
        yield sum((s_ngrams & r_ngrams).itervalues()) + 1
        yield max(len(hypothesis)+1-n, 0) + 1


def smooth_bleu(hypothesis, reference):
    stats = list(smooth_bleu_stats(hypothesis.split(), reference.split()))
    assert all(v != 0 for v in stats)
    (c, r) = stats[:2]
    log_bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)

try:
    import meteor_api
    meteor_api.initVM(maxheap='2g')
except ImportError:
    pass

METEOR_PARAPHRASE='/home/vchahune/projects/sp2013.11-731/hw4/meteor-1.4/data/paraphrase-en.gz'
class MeteorScorer:
    def __init__(self):
        logging.info('Loading the METEOR English scorer')
        config = meteor_api.MeteorConfiguration()
        config.setLanguage('en')
        config.setParaFileURL(meteor_api.URL('file:'+METEOR_PARAPHRASE))
        self.meteor_scorer = meteor_api.MeteorScorer(config)

    def stats(self):
        return MeteorStats()

    def score(self, hypothesis, reference):
        return MeteorStats(self.meteor_scorer.getMeteorStats(hypothesis, reference))

    def update(self, stats):
        self.meteor_scorer.computeMetrics(stats.stats)

class MeteorStats:
    def __init__(self, stats=None):
        if not stats:
            self.stats = meteor_api.MeteorStats()
        else:
            self.stats = stats

    def __iadd__(self, other):
        self.stats.addStats(other.stats)
        return self

    @property
    def score(self):
        return self.stats.score

    def __repr__(self):
        return self.stats.toString()

meteor_scorer = None
def meteor(hypothesis, reference):
    global meteor_scorer
    if not meteor_scorer: meteor_scorer = MeteorScorer()
    return meteor_scorer.score(hypothesis, reference).score

METRICS = {'bleu': smooth_bleu, 'meteor': meteor}
