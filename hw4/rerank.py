import argparse
import logging
import pickle
from itertools import groupby
from pro import read_nbest

all_observed_features = set()
def dot_product(fvector, weights):
    for fname in fvector:
        all_observed_features.add(fname)
    return sum(weights.get(fname, 0)*fval for fname, fval in fvector.iteritems())

def select_best(weights):
    return lambda t: dot_product(t[3], weights)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Re-rank n-best lists')
    parser.add_argument('weights', help='weight file to use for ranking')
    args = parser.parse_args()

    with open(args.weights) as f:
        weights = pickle.load(f)

    logging.info('Reranking with %d weights', len(weights))

    for sentence_id, group in groupby(read_nbest('/dev/stdin'), key=lambda t:t[0]):
        sentence_id, sentence, alignments, features = max(group, key=select_best(weights))
        print(sentence.encode('utf8'))

    logging.info('Observed %d features in test set', len(all_observed_features))
    all_weights = set(weights.iterkeys())
    if all_weights != all_observed_features:
        diff = (all_weights - all_observed_features) | (all_observed_features - all_weights)
        if len(diff) < 20:
            logging.info('Missing features: %s', ' '.join(diff))

if __name__ == '__main__':
    main()
