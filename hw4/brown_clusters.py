import logging
import gzip

SRC_CLUSTER_FILE = '/home/vchahune/projects/sp2013.11-731/hw4/data/ru-c600.gz'
TGT_CLUSTER_FILE = '/home/vchahune/projects/sp2013.11-731/hw4/data/en-c600.gz'
LM_FILE = '/home/vchahune/projects/sp2013.11-731/hw4/data/cluster-10gram.klm'

def read_clusters(filename):
    cluster = {}
    with gzip.open(filename) as f:
        for line in f:
            c, word, count = line.decode('utf8').split('\t')
            cluster[word] = c
    return cluster

logging.info('Reading clusters...')
SRC_CLUSTER = read_clusters(SRC_CLUSTER_FILE)
TGT_CLUSTER = read_clusters(TGT_CLUSTER_FILE)

def convert_source(word):
    return 'C'+SRC_CLUSTER.get(word, 'UNK')

def convert_target(word):
    return 'C'+TGT_CLUSTER.get(word, 'UNK')

import kenlm
LM = kenlm.LanguageModel(LM_FILE)
def lm_score(source, target, alignment):
    yield 'brown_lm_score', LM.score(' '.join(map(convert_target, target)))

from collections import Counter, defaultdict
def cluster_count(source, target, alignment):
    counts = Counter()
    for word in target:
        counts[convert_target(word)] += 1
    for cluster, count in counts.iteritems():
        yield cluster+'_count', count

TM_FILE = '/home/vchahune/projects/sp2013.11-731/hw4/code/data/corpus.ru-en.clus.model'
CLUSTER_TM = defaultdict(dict)
with open(TM_FILE) as f:
    for line in f:
        ru, en, prob = line.split()
        CLUSTER_TM[ru][en] = float(prob)

import numpy
def tm_score(source, target, alignment):
    log_prob = 0
    for en in target:
        log_prob += numpy.logaddexp.reduce(
            [CLUSTER_TM[convert_source(ru)].get(convert_target(en), -numpy.inf)
            for ru in source+['<eps>']])
    yield 'cluster_tm', log_prob
