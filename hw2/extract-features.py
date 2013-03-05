import sys
import json
from collections import defaultdict
import argparse
import editdist
import io
import nltk
from nltk.tokenize import word_tokenize

def get_ngrams(tokens, n):
  tokens = list(tokens)
  ngrams = []
  for i in range(0, n-1):
    tokens.append('</s>')
    tokens.insert(0, '<s>')
  for i in range(n-1, len(tokens)):   # e.g. in a bigram, i = 1..100
    ngram = ''
    for j in range(0, n):     # e.g. in a bigram, j = 0..1
      ngram = '{0}-{1}'.format(tokens[i-j], ngram)
    ngrams.append(ngram)
#  print 'tokens: {0}'.format(tokens)
#  print '{0}grams: {1}\n'.format(n, ngrams)
  return ngrams

# count matches
def ngram_matches(s1Tokens, s2Tokens, n):
  s1Ngrams = get_ngrams(s1Tokens, n)
  s2Ngrams = set(get_ngrams(s2Tokens, n))
  matches = 0
  for ngram in s1Ngrams:
    if ngram in s2Ngrams:
      matches += 1
#  print 's1: {0}'.format(s1Tokens)
#  print 's2: {0}'.format(s2Tokens)
#  print '{0}gram matches = {1}\n'.format(n, matches)
  return matches

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
    return score

# parse arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--inputFilename", type=str, help="input file, formatted as 'h1 ||| h2 ||| ref'")
argParser.add_argument("-f", "--featuresFilename", type=str, help="each line is a json-style dictionary of the features")
args = argParser.parse_args()
inputFile = open(args.inputFilename, mode='r')
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
  (h1, h2, ref) = (word_tokenize(h1Sent), word_tokenize(h2Sent), word_tokenize(refSent))
  (h1Set, h2Set, refSet) = ( set(h1), set(h2), set(ref) )

  # EXTRACT FEATURES
  features = {}

  # compute unigram precision diff
  h1_precision = 1.0 * unigram_matches(h1, refSet) / len(h1)
  h2_precision = 1.0 * unigram_matches(h2, refSet) / len(h2)
  features['BETTER-unigram-precision'] = h1_precision - h2_precision
  features['SAME-unigram-precision'] = abs( h1_precision - h2_precision )

  # compute unigram recall diff
  h1_recall = 1.0 * unigram_matches(ref, h1Set) / len(ref)
  h2_recall = 1.0 * unigram_matches(ref, h2Set) / len(ref)
  features['BETTER-unigram-recall'] = h1_recall - h2_recall
  features['SAME-unigram-recall'] = abs( h1_recall - h2_recall )

  # compute ngram precision diff
  # compute ngram recall diff
  for n in range(2, 5):
    h1_ngram_precision = 1.0 * ngram_matches(h1, ref, n) / (len(h1) + n - 1)
    h2_ngram_precision = 1.0 * ngram_matches(h2, ref, n) / (len(h2) + n - 1)
    features['BETTER-{0}gram-precision'.format(n)] = h1_ngram_precision - h2_ngram_precision
    features['SAME-{0}gram-precision'.format(n)] = abs(h1_ngram_precision - h2_ngram_precision)

    h1_ngram_recall = 1.0 * ngram_matches(ref, h1, n) / (len(h1) + n - 1)
    h2_ngram_recall = 1.0 * ngram_matches(ref, h2, n) / (len(h2) + n - 1)
    features['BETTER-{0}gram-recall'.format(n)] = h1_ngram_recall - h2_ngram_recall
    features['SAME-{0}gram-recall'.format(n)] = abs(h1_ngram_recall - h2_ngram_recall)

  # write features
  featuresFile.write(json.dumps(features))
  featuresFile.write('\n')

# close files
inputFile.close()
featuresFile.close()
