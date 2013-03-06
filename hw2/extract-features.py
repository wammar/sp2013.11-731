import sys
import json
from collections import defaultdict
import argparse
import editdist
import io
import nltk
from nltk.tokenize import word_tokenize
import math

NONE = -100
START = -500

typeToEmbeddings = defaultdict(list)
  
def read_embeddings(filename):
  eFile = open(filename)
  for line in eFile:
    (word, embeddings) = line.strip().split('\t')
    embeddings = embeddings.split()
    for i in range(0, len(embeddings)):
      embeddings[i] = float(embeddings[i])
    typeToEmbeddings[word] = embeddings

def embeddings_dist(type1, type2):
  if type1 not in typeToEmbeddings or type2 not in typeToEmbeddings:
    return None
  e1, e2 = typeToEmbeddings[type1], typeToEmbeddings[type2]
  assert(len(e1) == len(e2))
  distSquared = 0
  for i in range(0, len(e1)):
    distSquared += (e1[i] - e2[i])**2
  dist = math.sqrt(distSquared)
  return dist

def get_ngrams(tokens, n):
  tokens = list(tokens)
  ngrams = []
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

def tokens_are_similar(t1,t2):
  similarity = 1 - 1.0 * editdist.distance(t1, t2) / max(len(t1), len(t2))
  if similarity == 1 or (len(t1) > 3 and similarity > 0.5):
    return True

#  dist = embeddings_dist(t1, t2)
#  if dist < 0.001:
#    print 'embeddings think those are similar: {0}, {1}'.format(t1, t2)
#    return True
  
  return False

def approx_unigram_matches(h, ref):
  # align
  alignments = []
  for i in range(0, len(h)):
    alignment = NONE
    # note the src positions j may take are ordered in a clever way
    srcPosList = range(i, len(ref))
    temp = range(0, min(i, len(ref)))
    temp.reverse()
    srcPosList += temp
    for j in srcPosList:
      if tokens_are_similar(h[i], ref[j]):
#        print 'similar: {0}, {1}'.format(h[i], ref[j])
        alignment = j
        break
    alignments.append(alignment)
  assert(len(alignments) == len(h))
  # count the number of matches
  matches = sum(1 for a in alignments if a != NONE)
  # build a historgram of alignemnt jumps
  alignmentJumps = defaultdict(int)
  prev = START
  for i in range(0, len(alignments)):
    if alignments[i] >= 0:
      if prev >= 0:
        if alignments[i] - prev == 1:
#          assert(True)
          alignmentJumps[1] += 1
        elif abs(alignments[i] - prev) < 3:
#          assert(True)
          alignmentJumps[alignments[i] - prev] += 1
        else:
          lg = math.floor(math.log(abs(alignments[i] - prev)))
          sg = math.copysign(1, alignments[i] - prev)
          alignmentJumps['{0}exp{1}'.format(sg,lg)] += 1
      elif prev == START:
        if alignments[i] < 3:
          alignmentJumps[(START,alignments[i])] += 1
        else:
          alignmentJumps[(START,'>>0')] += 1
#      else:
#        alignmentJumps[(NONE,'>=0')] += 1
#    elif alignments[i] == NONE:
#      assert(True)
#      alignmentJumps[NONE] += 1
#    else: 
#      assert(False)
    prev = alignments[i]
  # normalize such that alignment jumps is the empirical distribution of jumps in this alignemnt
  for key in alignmentJumps:
    alignmentJumps[key] = 1.0 * alignmentJumps[key] / len(h)
#  print 'h: {0}'.format(h)
#  print 'ref: {0}'.format(ref)
#  print 'matches: {0}'.format(matches)
#  print 'jumps: {0}\n\n\n'.format(alignmentJumps)
  return (matches, alignmentJumps)
        
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

# read embeddings
#print 'reading-embeddings...'
#read_embeddings('data/word_embeddings.txt')
#print 'done.'

linesCounter = -1
for parallelLine in inputFile:

  linesCounter += 1
  if linesCounter % 1000 == 0:
    print '{0} lines read'.format(linesCounter)
  
  # parse sentence
  (h1Sent, h2Sent, refSent) = parallelLine.split('|||')
  (h1Char, h2Char, refChar) = (list(h1Sent), list(h2Sent), list(refSent))
  (h1, h2, ref) = (word_tokenize(h1Sent), word_tokenize(h2Sent), word_tokenize(refSent))
#  (h1, h2, ref) = (h1Sent.split(), h2Sent.split(), refSent.split())
  (h1Set, h2Set, refSet) = ( set(h1), set(h2), set(ref) )

  # EXTRACT FEATURES
  features = {}

  # compute unigram precision diff
#  h1_precision = 1.0 * unigram_matches(h1, refSet) / len(h1)
#  h2_precision = 1.0 * unigram_matches(h2, refSet) / len(h2)
#  features['BETTER-unigram-precision'] = h1_precision - h2_precision
#  features['SAME-unigram-precision'] = abs( h1_precision - h2_precision )

  # compute unigram recall diff
  h1_recall = 1.0 * unigram_matches(ref, h1Set) / len(ref)
  h2_recall = 1.0 * unigram_matches(ref, h2Set) / len(ref)
  features['BETTER-unigram-recall'] = h1_recall - h2_recall
#  features['SAME-unigram-recall'] = abs( h1_recall - h2_recall )

  # compute ngram precision diff
  # compute ngram recall diff
  for n in range(2, 6):
    if n == 4:
      continue
#    h1_ngram_precision = 1.0 * ngram_matches(h1, ref, n) / (len(h1) + n - 1)
#    h2_ngram_precision = 1.0 * ngram_matches(h2, ref, n) / (len(h2) + n - 1)
#    features['BETTER-{0}gram-precision'.format(n)] = h1_ngram_precision - h2_ngram_precision
#    features['SAME-{0}gram-precision'.format(n)] = abs(h1_ngram_precision - h2_ngram_precision)

    h1_ngram_recall = 1.0 * ngram_matches(ref, h1, n) / (len(h1) + n - 1)
    h2_ngram_recall = 1.0 * ngram_matches(ref, h2, n) / (len(h2) + n - 1)
    features['BETTER-{0}gram-recall'.format(n)] = h1_ngram_recall - h2_ngram_recall
#    features['SAME-{0}gram-recall'.format(n)] = abs(h1_ngram_recall - h2_ngram_recall)

  # compute character ngram recall and precision
  for n in [3, 9, 15]:
    h1_char_ngram_recall = 1.0 * ngram_matches(refChar, h1Char, n) / (len(h1Char) + n - 1)
    h2_char_ngram_recall = 1.0 * ngram_matches(refChar, h2Char, n) / (len(h2Char) + n - 1)
    features['BETTER-char-{0}gram-recall'.format(n)] = h1_char_ngram_recall - h2_char_ngram_recall

  # approx matches and alignments
  # approx precision
#  (h1ApproxMatches, h1AlignmentJumps) = approx_unigram_matches(h1, ref)
#  (h2ApproxMatches, h2AlignmentJumps) = approx_unigram_matches(h2, ref)
#  h1_approx_precision = 1.0 * h1ApproxMatches / len(h1)
#  h2_approx_precision = 1.0 * h2ApproxMatches / len(h2)
#  features['BETTER-approx-precision'] = h1_approx_precision - h2_approx_precision
#  features['SAME-approx-precision'] = abs(h1_approx_precision - h2_approx_precision)
  # approx recall
  (h1ReverseApproxMatches, dummy) = approx_unigram_matches(ref, h1)
  (h2ReverseApproxMatches, dummy) = approx_unigram_matches(ref, h2)
  h1_approx_recall = 1.0 * h1ReverseApproxMatches / len(ref)
  h2_approx_recall = 1.0 * h2ReverseApproxMatches / len(ref)
  features['BETTER-approx-recall'] = h1_approx_recall - h2_approx_recall
#  features['SAME-approx-recall'] = abs(h1_approx_recall - h2_approx_recall)
  # alignment jump diff between h1 and h2
#  for key in h1AlignmentJumps.keys():
#    features['BETTER-jump-{0}'.format(key)] = h1AlignmentJumps[key] - h2AlignmentJumps[key]
#    features['SAME-jump-{0}'.format(key)] = abs(h1AlignmentJumps[key] - h2AlignmentJumps[key])
#  for key in h2AlignmentJumps.keys():
#    if key not in h1AlignmentJumps:
#      features['BETTER-jump-{0}'.format(key)] = h1AlignmentJumps[key] - h2AlignmentJumps[key]
#      features['SAME-jump-{0}'.format(key)] = abs(h1AlignmentJumps[key] - h2AlignmentJumps[key])
  # alignment jump
#  for key in h1AlignmentJumps.keys():
#    features['BETTER-h1-jump-{0}'.format(key)] = h1AlignmentJumps[key]  
#  for key in h2AlignmentJumps.keys():
#    features['BETTER-h2-jump-{0}'.format(key)] = h2AlignmentJumps[key]
 
  # brevity penalty
  features['BETTER-length-diff'] = 1.0 * abs(len(h1) - len(ref)) / max(len(h1), len(ref)) -  1.0 * abs(len(h2) - len(ref)) / max(len(h1), len(ref))
#  features['BETTER-length**2-diff'] = (len(h1) - len(ref))**2 - (len(h2) - len(ref))**2

  # 
 
  # write features
  featuresFile.write(json.dumps(features))
  featuresFile.write('\n')

# close files
inputFile.close()
featuresFile.close()
