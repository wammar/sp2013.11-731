import io
import argparse
from itertools import izip, groupby
from pro import read_nbest

def read_source(filename):
    with io.open(filename, encoding='utf8') as f:
        for line in f:
            yield line.strip().split('|||')[1]

def extract_features(source, target, alignment, config):
    source_words = source.split()
    target_words = target.split()
    return dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source_words, target_words, alignment))

def add_features(features, init_features):
    features.update(init_features)
    return ' '.join(u'{}={}'.format(k, v) for k, v in features.iteritems())

def main():
    parser = argparse.ArgumentParser(description='Add features to n-best list')
    parser.add_argument('nbest', help='n-best file')
    parser.add_argument('align', help='alignments file')
    parser.add_argument('source', help='source file')
    parser.add_argument('config', help='config file')
    parser.add_argument('-features_only', type=bool, default=False, help='print features only')
    args = parser.parse_args()

    config = __import__(args.config)

    nbests = groupby(read_nbest(args.nbest, args.align), key=lambda x: x[0])
    for (_, group), source in izip(nbests, read_source(args.source)):
        for sid, target, alignments, init_features in group:
            features = add_features(extract_features(source, target, alignments, config), init_features)
            if args.features_only:
                for f in features.split(' '):
                    print f.split('=')[0].encode('utf8')
            else:
                print(u'{} ||| {} ||| {}'.format(sid, target, features).encode('utf8'))

if __name__ == '__main__':
    main()
