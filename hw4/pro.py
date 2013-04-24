import io
import logging
import heapq
import argparse
import pickle
import random
import math
import scipy
from collections import defaultdict, namedtuple
from itertools import groupby, izip
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
import metrics

Candidate = namedtuple('Candidate', 'score, alignments, features')

def read_nbest(nbest_filename, align_filename=None):
    """returns a list of (id, ['tokenized', 'sentence'], [(0, 0), (1, 2)], {f1:v1, f2:v2})"""
    af = io.open(align_filename, encoding='utf8') if align_filename != None else None
    with io.open(nbest_filename, encoding='utf8') as f:
        for line in f:
            aline = af.readline() if af else None
            sentence_id, sentence, features = line.strip().split(' ||| ')
            alignments = []
            if aline:
                alignments_str = aline.strip().split()
                for alignment_str in alignments_str:
                    (src_pos, tgt_pos) = alignment_str.split('-')
                    alignments.append( (int(src_pos), int(tgt_pos)) )
            features = dict((k, float(v)) for k, v in
                            (kv.split('=') for kv in features.split()))
            yield (sentence_id, sentence, alignments, features)


def read_ref(filename):
    with io.open(filename, encoding='utf8') as f:
        for line in f:
            yield line.strip()


def read_candidates(nbest_filename, ref_filename, align_filename, scorer):
    """returns a dictionary that maps sentence id to a list of Candidate"""
    # Compute scores for all hypothesis/reference paris
    nbests = groupby(read_nbest(nbest_filename, align_filename), key=lambda x: x[0])
    refs = read_ref(ref_filename)
    candidates = defaultdict(list)
    for ((k, group), ref) in izip(nbests, refs):
        for (_, hyp, alignments, features) in group:
            candidate = Candidate(scorer(hyp, ref), alignments, features)
            candidates[k].append(candidate)

    # Convert feature vectors to scipy sparse vectors
    vectorizer = DictVectorizer()
    vectorizer.fit([candidate.features for candidate_list in candidates.itervalues()
        for candidate in candidate_list])

    for sentence_id, candidate_list in candidates.iteritems():
        vectorized_candidates = []
        for candidate in candidate_list:
            features = vectorizer.transform(candidate.features)
            vectorized_candidates.append(Candidate(candidate.score, candidate.alignments, features))
        candidates[sentence_id] = vectorized_candidates

    return vectorizer, candidates


def get_pairs(candidate_list, n_samples, n_pairs, score_threshold):
    pairs = []
    for _ in range(n_samples):
        ci = candidate_list[random.randrange(0, len(candidate_list))]
        cj = candidate_list[random.randrange(0, len(candidate_list))]
        if abs(ci.score - cj.score) < score_threshold: continue
        pairs.append((ci.features - cj.features, ci.score - cj.score))
    for x, y in heapq.nlargest(n_pairs, pairs, key=lambda xy: abs(xy[1])):
        yield x, y
        yield -x, -y


REF = '/home/vchahune/projects/sp2013.11-731/hw4/data/dev.ref'

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='PRO tuning')
    parser.add_argument('--nbest', help='n-best file', default='/dev/stdin')
    parser.add_argument('--ref', help='reference file', default=REF)
    parser.add_argument('--align', help='alignments file')
    parser.add_argument('--metric', help='metric to use for tuning',
            choices=metrics.METRICS.keys(), default='bleu')
    parser.add_argument('--n_samples', help='number of samples', type=int, default=5000)
    parser.add_argument('--n_pairs', help='max number of pairs', type=int, default=50)
    parser.add_argument('--threshold', help='score threshold', type=float, default=0.05)
    parser.add_argument('--weights', help='file path to store weights to')
    parser.add_argument('--l1', help='L1 regularization penalty', type=float, default=100.)
    args = parser.parse_args()

    logging.info('Reading all candidates and computing features')

    vectorizer, candidates = read_candidates(args.nbest,
            args.ref, args.align, metrics.METRICS[args.metric])

    logging.info('Read %d candidates with %d features', len(candidates),
            len(vectorizer.get_feature_names()))
    X, Y = [], []
    logging.info('Producing training data')
    for candidate_list in candidates.itervalues():
        for x, y in get_pairs(candidate_list, args.n_samples, args.n_pairs, args.threshold):
            X.append(x)
            Y.append(y)
    X = scipy.sparse.vstack(X)
    logging.info('Fitting the model (%d instances, %d features)', *X.shape)
    logging.info('L1 penalty: %s', args.l1)
    model = Ridge(fit_intercept=False, alpha=args.l1)
    model.fit(X, Y)

    w = model.coef_
    w /= math.sqrt((w**2).sum() + 1e-6) # normalize weight vector
    weights = dict(zip(vectorizer.get_feature_names(), w))

    score = sum(max(candidate_list, key=lambda c: c.features.dot(w)).score
            for candidate_list in candidates.itervalues())/len(candidates)

    print '\n\n\n'
    print(u'Most +ve features: {}'.format(' '.join(u'{}={:.2f}'.format(*kv)
        for kv in heapq.nlargest(50, weights.iteritems(), key=lambda t: t[1]))).encode('utf8'))
    print(u'Most -ve features: {}'.format(' '.join(u'{}={:.2f}'.format(*kv)
        for kv in heapq.nsmallest(50, weights.iteritems(), key=lambda t: t[1]))).encode('utf8'))
    print '\n\n\n'
    print('Dev score: {}'.format(score))
    print '==END-OF-PRO=='
    
    if args.weights:
        with open(args.weights, 'w') as weights_file:
            pickle.dump(weights, weights_file)

if __name__ == '__main__':
    main()
