import sys
import json
import argparse
import creg
from collections import defaultdict

sentLevelFeatures = defaultdict(lambda:defaultdict(lambda:{}))

def ReadExampleFeatures(files):
  # read the features
  features = {}
  endOfFile = False
  for i in range(0,len(files)):
    line = files[i].readline()
    if(len(line) == 0):
      endOfFile = True
      break;
    obj = json.loads(line)
    for key in obj.keys():
      # verify the id is the same across all train feature files
      if key == 'id' and 'id' in features:
        if obj['id'] != features['id']:
          print obj['id']
          print features['id']
          assert(False)
      else:
        features[key] = obj[key]
  if endOfFile:
    features = None
  else:
    # combine some features
    features['m1-fwd&bwd'] = 1 if features['m1-fwd'] == 1 and features['m1-bwd'] == 1 else 0
    features['m1-fwd|bwd'] = 1 if features['m1-fwd'] == 1 or features['m1-fwd'] == 1 else 0
    features['prevTgtPos:srcPos'] = 1 if features['tgtPos'] > 0 and sentLevelFeatures[features['tgtPos']-1][features['srcPos']]['m1-fwd&bwd'] == 1 else 0
    features['tgtPos:prevSrcPos'] = 1 if features['srcPos'] > 0 and sentLevelFeatures[features['tgtPos']][features['srcPos']-1]['m1-fwd&bwd'] == 1 else 0
    features['prevTgtPos:prevSrcPos'] = 1 if features['tgtPos']*features['srcPos'] > 0 and sentLevelFeatures[features['tgtPos']-1][features['srcPos']-1]['m1-fwd&bwd'] == 1 else 0
    sentLevelFeatures[features['tgtPos']][features['srcPos']] = features
    features['noway'] = 1 if features['tgtPos'] > 0 \
        and features['srcPos'] > 0 \
        and features['m1-fwd|bwd'] == 0 \
        and sentLevelFeatures[features['tgtPos']][features['srcPos']-1]['m1-fwd|bwd'] == 0 \
        and sentLevelFeatures[features['tgtPos']-1][features['srcPos']]['m1-fwd|bwd'] == 0 \
        and sentLevelFeatures[features['tgtPos']-1][features['srcPos']-1]['m1-fwd|bwd'] == 0 \
        else 0
  return features;

def ReadResponse(originalFile):
  line = originalFile.readline()
  if(len(line) == 0):
    return None
  obj = json.loads(line)
  return int(obj['Score'])

def CreateDataset(trainFeaturesFiles, trainResponseFile):
  # create training set
  trainData = []
  while True:
    # read features
    features = ReadExampleFeatures(trainFeaturesFiles)
    if features == None:
      break

    # read response
    response = ReadResponse(trainResponseFile)

    # append to training data
    trainData.append((features, response))

  return trainData

# parse arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-tf", "--trainFeaturesFilenames", type=str, help="comma-separated training features filenames")
argParser.add_argument("-ef", "--evalFeaturesFilenames", type=str, help="comma-separated evaluation features filenames")
argParser.add_argument("-tr", "--trainResponseFilename", type=str, help="comma-separated training response filename")
argParser.add_argument("-er", "--evalResponseFilename", type=str, help="comma-separated evaluation response filename")
argParser.add_argument("-ao", "--outputAlignmentsFilename", type=str, help="output alignments")
args = argParser.parse_args()
trainFeaturesFilenames = args.trainFeaturesFilenames.split(',')
evalFeaturesFilenames = args.evalFeaturesFilenames.split(',')
trainResponseFilename = args.trainResponseFilename
evalResponseFilename = args.evalResponseFilename

# open files
trainFeaturesFiles, evalFeaturesFiles = [], []
for trainFeaturesFilename in trainFeaturesFilenames:
  f = open(trainFeaturesFilename, 'r')
  trainFeaturesFiles.append( f ) 
for evalFeaturesFilename in evalFeaturesFilenames:
  f = open(evalFeaturesFilename, 'r')
  evalFeaturesFiles.append( f )
trainResponseFile, evalResponseFile = open(trainResponseFilename, 'r'), open(evalResponseFilename, 'r')

# create train dataset
trainDatasetRaw = CreateDataset(trainFeaturesFiles, trainResponseFile)
# pack it in creg.RealvaluedDataset
trainDataset = creg.CategoricalDataset(trainDatasetRaw)

# train the model
model = creg.LogisticRegression(l2=1.0)
model.fit(trainDataset)
print 'model weights:'
print model.weights

# create eval dataset
evalDatasetRaw = CreateDataset(evalFeaturesFiles, evalResponseFile)
# pack it in creg.RealvaluedDataset
evalDataset = creg.CategoricalDataset(evalDatasetRaw)

# evaluate
dataset = trainDatasetRaw
dataset.extend(evalDatasetRaw)
trainPredictions = model.predict(trainDataset)
evalPredictions = model.predict(evalDataset)
predictions = []
for prediction in trainPredictions:
  predictions.append(prediction)
for prediction in evalPredictions:
  predictions.append(prediction)
truth = []
for (x,y) in dataset:
  truth.append(y)
vs = zip(predictions, truth)
#errors = sum( abs(pred - real) for (pred, real) in vs)
falsePositives, falseNegatives, truePositives, trueNegatives = 0, 0, 0, 0
for (pred, real) in vs:
  if pred == 1 and real == 0:
    falsePositives += 1
  elif pred == 0 and real == 1:
    falseNegatives += 1
  elif pred == 1 and real == 1:
    truePositives += 1
  elif pred == 0 and real == 0:
    trueNegatives += 1
  elif real == None:
    break
  else:
    print pred, real
    assert(False)

precision = truePositives * 1.0 / (truePositives + falsePositives)
print 'precision = {0}'.format(precision)
recall = truePositives * 1.0 / (truePositives + falseNegatives)
print 'recall = {0}'.format(recall)
accuracy =  1.0 * (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
print 'accuracy = {0}'.format(accuracy)
f1 = 2.0 * precision * recall / (precision + recall)
print 'f1 = {0}'.format(f1)

# convert the predictions into moses-style alignments
outputAlignmentsFile = open(args.outputAlignmentsFilename, 'w')

currentSentId = 0
for i in range(0, len(predictions)):
  (features, dummyResponse) = dataset[i]
  pred = predictions[i]
  (sentId, tgtPos, srcPos) = features['sentId'], features['tgtPos'], features['srcPos']
  if sentId < currentSentId or sentId > currentSentId+1:
    assert(False)
  elif sentId != currentSentId:
    outputAlignmentsFile.write('\n')
    currentSentId += 1
  if pred == 1:
    outputAlignmentsFile.write('{0}-{1} '.format(srcPos, tgtPos))
  elif pred == 0:
    pred = 0
  elif pred == None:
    break
  else:
    print pred
    assert(False)

outputAlignmentsFile.write('\n')
outputAlignmentsFile.close()
