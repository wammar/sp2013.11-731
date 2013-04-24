import argparse
import sys
import logging
from itertools import izip
import metrics

METRICS = {'meteor': metrics.MeteorScorer}

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Score outputs')
    parser.add_argument('metric', choices=METRICS.keys())
    parser.add_argument('reference')
    args = parser.parse_args()

    scorer = METRICS[args.metric]()
    aggregate_stats = scorer.stats()
    with open(args.reference) as ref:
        for hypothesis, reference in izip(sys.stdin, ref):
            aggregate_stats += scorer.score(hypothesis, reference)

    scorer.update(aggregate_stats)
    print('{}: {:.5f}'.format(args.metric.upper(), aggregate_stats.score*100))

if __name__ == '__main__':
    main()

