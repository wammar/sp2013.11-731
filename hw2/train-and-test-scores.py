import sys
import json
import argparse
import creg
from collections import defaultdict

SAME = 'SAME'
INVALID = 'INVALID'
BETTER = 'BETTER'

#sentLevelFeatures = defaultdict(lambda:defaultdict(lambda:{}))

def ReadExampleFeatures(featuresFile):
  # read the features
  line = featuresFile.readline()
  if(len(line) == 0):
    features = None
  else:
    features = json.loads(line)
  return features;

def ReadResponse(originalFile, classifierType):
  line = originalFile.readline()
  if(len(line) == 0):
    return None

  if classifierType == SAME:
    response = SAME if line.strip() == '0' else 'NOT-{0}'.format(SAME)

  elif classifierType == BETTER:
    if line.strip() == '0':
      response = INVALID
    elif line.strip() == '1':
      response = BETTER
    else:
      response = 'NOT-{0}'.format(BETTER)
  return response

def CreateDataset(trainFeaturesFile, trainResponseFile, classifierType):
  # create training set
  trainData = []
  while True:
    # read features
    features = ReadExampleFeatures(trainFeaturesFile)
    if features == None:
      break

    # read response
    response = ReadResponse(trainResponseFile, classifierType)

    # append to training data
    if response != INVALID:
      continue
    else:
      trainData.append((features, response))

  return trainData

def TranslateLabelsIntoAnswer(samePrediction, betterPrediction):
  assert(same in [SAME, 'NOT-{0}'.format(SAME)] and better in [BETTER, 'NOT-{0}'.format(BETTER), INVALID])
  if same == SAME:
    return '0'
  else:
    if better == BETTER:
      return '1'
    else:
      return '-1'  

# parse arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-tf", "--trainFeaturesFilename", type=str, help="input -- features of the training set, each line is formatted as a json dictionary")
argParser.add_argument("-tr", "--trainResponseFilename", type=str, help="input -- response of the training set, each line is 0, -1 or 1")
argParser.add_argument("-sf", "--testFeaturesFilename", type=str, help="input -- features of the test set, each line is formatted as a json dictionary")
argParser.add_argument("-sr", "--testResponseFilename", type=str, help="(optional) input -- response of the test set, each line is 0, -1 or 1", default="")
argParser.add_argument("-pr", "--predictedResponseFilename", type=str, help="(optional) output -- response predicted according to the model for the test set, each line is 0, -1, or 1", default="")
args = argParser.parse_args()
trainFeaturesFile = open(args.trainFeaturesFilename, 'r')
trainResponseFile = open(args.trainResponseFilename, 'r')
testFeaturesFile = open(args.testFeaturesFilename, 'r')
testResponseFile = open(args.testResponseFilename, 'r') if len(args.testResponseFilename) > 0 else None
predictedResponseFile = open(args.predictedResponseFilename, 'w') if len(args.predictedResponseFilename) > 0 else None

# TRAIN 'SAME' CLASSIFIER
# create train dataset
trainDatasetRaw = CreateDataset(trainFeaturesFile, trainResponseFile, SAME)
# pack it in creg.RealvaluedDataset
trainDataset = creg.CategoricalDataset(trainDatasetRaw)
# train the model
sameModel = creg.LogisticRegression(l2=1.0)
sameModel.fit(trainDataset)
print 'same model weights:'
print sameModel.weights

# TRAIN 'BETTER' CLASSIFIER
trainDatasetRaw = CreateDataset(trainFeaturesFile, trainResponseFile, BETTER)
trainDataset = creg.CategoricalDataset(trainDatasetRaw)
betterModel = creg.LogisticRegression(l2=1.0)
betterModel.fit(trainDataset)
print 'better model weights:'
print betterModel.weights

# create eval datasets
evalSameDatasetRaw = CreateDataset(testFeaturesFile, testResponseFile, SAME)
evalSameDataset = creg.CategoricalDataset(evalBetterDatasetRaw)
evalBetterDatasetRaw = CreateDataset(testFeaturesFile, testResponseFile, BETTER)
evalBetterDataset = creg.CategoricalDataset(evalBetterDatasetRaw)

# evaluate
evalSamePredictions = sameModel.predict(evalSameDataset)
evalBetterPredictions = betterModel.predict(evalBetterDataset)
samePredictions, betterPredictions, predictions = [], [], []
for same, better in zip(evalSamePredictions, evalBetterPredictions):
  output = TranslateLabelsIntoAnswer(same, better)
  predictions.append(output)
  samePredictions.append(same)
  betterPredictions.append(INVALID if same == SAME else better)
  if predictedResponseFile:
    predictedResponseFile.write(output + '\n')

# analyze
if testResponseFile != None:
  sameMatrix = defaultdict(int)
  for (pred, (f, truth)) in zip(samePredictions, evalSameDataset):
    sameMatrix['{0}, {1}'.format(pred, truth)] += 1
  print '\n\nSAME MATRIX:\n'
  for key,val in sameMatrix:
    print '{0} => {1}'.format(key, val)
  betterMatrix = defaultdict(int)
  for (pred, (f, truth)) in zip(betterPredictions, evalBetterDataset):
    betterMatrix['{0}, {1}'.format(pred, truth)] += 1
  print '\n\nBETTER MATRIX:\n'
  for key,val in betterMatrix:
    print '{0} => {1}'.format(key, val)

trainFeaturesFile.close()
trainResponseFile.close()
testFeaturesFile.close()
if testResponseFile:
  testResponseFile.close()
if predictedResponseFile:
  predictedResponseFile.close() 
