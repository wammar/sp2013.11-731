#--encoding:utf-8--
import numpy
import cPickle
import math

VOC_FILE = '/usr0/home/mfaruqui/mt/4/data/vocab.devsrc.testsrc'
feature_vocabulary = {}
with open(VOC_FILE) as f:
    for line in f:
        word, freq = line.strip().split()
        feature_vocabulary[word] = len(feature_vocabulary)

MODEL_FILE = '/usr1/home/vchahune/data/dwl-model.pickle'
with open(MODEL_FILE) as f:
    models = cPickle.load(f)

def get_dwl_prob(source, target, alignment):
    X = numpy.zeros(len(feature_vocabulary))
    for word in source:
        if word in feature_vocabulary:
            X[feature_vocabulary[word]] += 1

    sent_log_prob = 0
    for word in target:
        if not word in models: continue
        clf = models[word]
        probs = clf.predict_proba(X)
        sent_log_prob += math.log(probs[0,1]/probs[0,0])
    yield 'dwl', sent_log_prob

if __name__=='__main__':
    print next(get_dwl_prob('я был единственным , кто занялся копированием демо на кассете .'.split(), 'admitted adopt adoption .'.split(), []))
