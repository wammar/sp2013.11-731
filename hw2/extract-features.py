import sys
import json
from collections import defaultdict
import argparse
import editdist
import io

# count unigram matches
def unigram_matches(h, ref):
    return sum(1 for w in h if w in ref)

def score(prec, recall, recall_weight):
#  return prec
  assert(recall_weight >= 0 and recall_weight <= 1)
  prec_weight = 1 - recall_weight
  if prec+recall == 0:
    return 0
  else:
    score = prec * recall / (prec_weight * prec + recall_weight * recall)
#    score = prec_weight * prec + recall_weight * recall
    return score

# parse arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--inputFilename", type=str, help="input file, formatted as 'h1 ||| h2 ||| ref'")
argParser.add_argument("-f", "--featuresFilename", type=str, help="each line is a json-style dictionary of the features")
args = argParser.parse_args()
inputFile = io.open(args.inputFilename, encoding='utf8', mode='r')
featuresFile = open(args.featuresFilename, mode='w')

# punctuations list
punc = set([',', '.', '!', ';', '?', '\'', '"', '(', ')', '-', '>', '<', '`', '=', '+', '#', '@'])

linesCounter = -1
for parallelLine in inputFile:

  linesCounter += 1
  if linesCounter % 1000 == 0:
    print '{0} lines read'.format(linesCounter)
  
  # parse sentence
  (h1Sent, h2Sent, refSent) = parallelLine.split('|||')
  (h1, h2, ref) = (h1Sent.split(), h2Sent.split(), refSent.split())
  (h1Set, h2Set, refSet) = ( set(h1), set(h2), set(ref) )

  # EXTRACT FEATURES
  features = {}

  # compute unigram precision diff
  h1_precision = 1.0 * unigram_matches(h1, refSet) / len(h1)
  h2_precision = 1.0 * unigram_matches(h2, refSet) / len(h2)
  features['unigram-precision'] = h1_precision - h2_precision

  # compute unigram recall diff
  h1_recall = 1.0 * unigram_matches(ref, h1Set) / len(ref)
  h2_recall = 1.0 * unigram_matches(ref, h2Set) / len(ref)
  features[u'unigram-recall'] = h1_recall - h2_recall

  # compute diff in harmonic mean of unigram precision,recall
  recall_weight = 0.7
  h1_harmonic = score(h1_precision, h1_recall, recall_weight)
  h2_harmonic = score(h2_precision, h2_recall, recall_weight)
  features['harmonic'] = h1_harmonic - h2_harmonic

  # write features
  featuresFile.write(json.dumps(features))
  featuresFile.write('\n')
      
# close files
inputFile.close()
featuresFile.close()
