import sys
import json
from collections import defaultdict
import argparse

import editdist

def ParseAlignmentsLine(line, bwd, tgtLength, srcLength):
  alignments = defaultdict(set)
  alignmentPoints = line.split()
  for point in alignmentPoints:
    if '-' in point:
      splitter = '-'
    else:
      splitter = '?'
    if bwd:
      (tgtPos, srcPos) = point.split(splitter)
    else:
      (srcPos, tgtPos) = point.split(splitter)
    tgtPos = int(tgtPos)
    assert(tgtPos < tgtLength)
    srcPos = int(srcPos)
    assert(srcPos < srcLength)
    alignments[tgtPos].add(srcPos)
  return alignments

# parse arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-st", "--parallelCorpusFilename", type=str, help="parallel corpus file ala src|||tgt ")
argParser.add_argument("-af", "--alignmentsFilename", type=str, help="a moses-style unsupervised alignments file")
argParser.add_argument("-an", "--alignmentsName", type=str, help="an identifier for the algorithm that resulted in these alignments")
argParser.add_argument("-bwd", "--alignmentsAreBackward", type=bool, help="a moses-style alignments file", default=False)
argParser.add_argument("-from", "--fromLineNumber", type=int, help="read from this line in the alignments file", default=0)
argParser.add_argument("-evalFrom", "--evaluateFromLineNumber", type=int, help="read to (excluding) this line in the alignments file", default=100)
argParser.add_argument("-to", "--toLineNumber", type=int, help="read to (excluding) this line in the alignments file", default=300)
argParser.add_argument("-ex", "--extractExtraFeatures", type=bool, help="extract extra features (e.g. positional)", default=True)
argParser.add_argument("-gf", "--goldAlignmentsFilename", type=str, help="a moses-style gold alignments file", default="")
argParser.add_argument("-ff", "--featuresFilename", type=str, help="extracted training example features")
argParser.add_argument("-rf", "--responseFilename", type=str, help="extracted training example labels")
args = argParser.parse_args()
parallelCorpusFile = open(args.parallelCorpusFilename, 'r')
alignmentsFile = open(args.alignmentsFilename, 'r')
if args.goldAlignmentsFilename > 0: 
  goldAlignmentsFile = open(args.goldAlignmentsFilename, 'r')
featuresFile = open(args.featuresFilename + '.train', 'w')
if args.responseFilename > 0:
  responseFile = open(args.responseFilename + '.train', 'w')

linesCounter = -1
for parallelLine in parallelCorpusFile:
  # read lines
  linesCounter += 1
  if linesCounter >= args.toLineNumber:
    break
  alignmentsLine = alignmentsFile.readline()
  if args.goldAlignmentsFilename:
    goldAlignmentsLine = goldAlignmentsFile.readline()
  if linesCounter < args.fromLineNumber:
    continue

  if linesCounter == args.evaluateFromLineNumber:
    featuresFile.close()
    featuresFile = open(args.featuresFilename + '.eval', 'w')
    if args.responseFilename > 0:
      responseFile.close()
      responseFile = open(args.responseFilename + '.eval', 'w')

  # parse sentence
  srcSent, tgtSent = parallelLine.split('|||')
  srcTokens = srcSent.split()
  tgtTokens = tgtSent.split()
  alignments = ParseAlignmentsLine(alignmentsLine, args.alignmentsAreBackward, len(tgtTokens), len(srcTokens))
  if args.goldAlignmentsFilename:
    gold = ParseAlignmentsLine(goldAlignmentsLine, False, len(tgtTokens), len(srcTokens))
    
  # write a training example for each point on the alignments grid
  for tgtPos in range(0, len(tgtTokens)):
    for srcPos in range(0, len(srcTokens)):
      features = {}
      # identifier of this training example
      #features['id'] = '{0}-{1}-{2}'.format(linesCounter, tgtPos, srcPos)
      features['sentId'] = linesCounter
      features['tgtPos'] = tgtPos
      features['srcPos'] = srcPos
      # unsupervised alignment
      if srcPos in alignments[tgtPos]:
        features[args.alignmentsName] = 1.0
      else:
        features[args.alignmentsName] = 0.0
      # extra features
      features['positional'] = abs(1.0 * (tgtPos+1) / len(tgtTokens) - (srcPos+1) / len(srcTokens))
      features['edit_distance'] = 1.0 * editdist.distance(tgtTokens[tgtPos], srcTokens[srcPos]) / max(len(tgtTokens[tgtPos]), len(srcTokens[srcPos]))
      features['capital'] = 1 if tgtTokens[tgtPos][0] >= 'A' and tgtTokens[tgtPos][0] <= 'Z' and srcTokens[srcPos][0] >= 'A' and srcTokens[srcPos][0] <= 'Z' and srcPos > 0 and tgtPos > 0 else 0;

      # write features
      featuresFile.write(json.dumps(features))
      featuresFile.write('\n')
      
      # response
      if args.goldAlignmentsFilename and goldAlignmentsLine:
        response = 1 if srcPos in gold[tgtPos] else 0
        responseFile.write(json.dumps({'Score':response}))
        responseFile.write('\n')
      
# close files
parallelCorpusFile.close()
alignmentsFile.close()
if args.goldAlignmentsFilename: 
  goldAlignmentsFile.close()
featuresFile.close()
if args.responseFilename:
  responseFile.close()
