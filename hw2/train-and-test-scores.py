import sys
import json
import argparse
import creg
from collections import defaultdict

SAME = 'SAME'
INVALID = 'INVALID'
BETTER = 'BETTER'
UNKNOWN = 'UNKNOWN'

#sentLevelFeatures = defaultdict(lambda:defaultdict(lambda:{}))

def ReadExampleFeatures(featuresFile, classifierType):
  assert(classifierType == 'SAME' or classifierType == 'BETTER')
  # read the features
  line = featuresFile.readline()
  if(len(line) == 0):
    return None
  else:
    features = json.loads(line)
  for key in features.keys():
    if not key.startswith(classifierType):
      features[key] = 0.0
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

def CreateDataset(featuresFile, responseFile, classifierType, include_invalid_sents=False):
  if responseFile:
    print 'creating a dataset for a {0} classifier. from file features file {1} and response file {2}...'.format(classifierType, featuresFile.name, responseFile.name)
  else:
    print 'creating a dataset for a {0} classifier. from file features file {1}...'.format(classifierType, featuresFile.name)
  # create training set
  trainData = []
  while True:
    # read features
    features = ReadExampleFeatures(featuresFile, classifierType)
    if features == None:
      break

    # read response
    if responseFile:
      response = ReadResponse(responseFile, classifierType)
    else:
      response = UNKNOWN

    # append to training data
    if response == INVALID and not include_invalid_sents:
      continue
    else:
      trainData.append((features, response))

  # rewind the files for future use
  featuresFile.seek(0)
  if responseFile:
    responseFile.seek(0)
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
trainSameDatasetRaw = CreateDataset(trainFeaturesFile, trainResponseFile, SAME)
print 'SAME training dataset contains {0} examples.'.format(len(trainSameDatasetRaw))
# pack it in creg.RealvaluedDataset
trainSameDataset = creg.CategoricalDataset(trainSameDatasetRaw)
# train the model
sameModel = creg.LogisticRegression(l2=100.0)
print 'started fitting the SAME model...'
sameModel.fit(trainSameDataset)
print 'done. same model weights:'
print sameModel.weights
print '\n\n'

# TRAIN 'BETTER' CLASSIFIER
trainBetterDatasetRaw = CreateDataset(trainFeaturesFile, trainResponseFile, BETTER)
print 'BETTER training dataset contains {0} examples.'.format(len(trainBetterDatasetRaw))
trainBetterDataset = creg.CategoricalDataset(trainBetterDatasetRaw)
betterModel = creg.LogisticRegression(l2=10.0)
print 'started fitting the BETTER model...'
betterModel.fit(trainBetterDataset)
print 'done. better model weights:'
print betterModel.weights
print '\n\n'

print '============== TRAINING FINISHED! ==============\n\n'

# create eval datasets
evalSameDatasetRaw = CreateDataset(testFeaturesFile, testResponseFile, SAME)
print 'SAME eval dataset contains {0} examples.'.format(len(evalSameDatasetRaw))
evalSameDataset = creg.CategoricalDataset(evalSameDatasetRaw)
evalBetterDatasetRaw = CreateDataset(testFeaturesFile, testResponseFile, BETTER, include_invalid_sents=True)
print 'BETTER eval dataset contains {0} examples.'.format(len(evalBetterDatasetRaw))
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
if testResponseFile:

  # build same matrix
  sameMatrix = defaultdict(int)
  for (pred, (f, truth)) in zip(samePredictions, evalSameDataset):
    sameMatrix['{0}, {1}'.format(pred, truth)] += 1

  # print same matrix
  print '\n\nSAME MATRIX:\n'
  for key in sameMatrix.keys():
    val = sameMatrix[key]
    print '{0} => {1}'.format(key, val)

  # build better matrix
  betterMatrix = defaultdict(int)
  for (pred, (f, truth)) in zip(betterPredictions, evalBetterDataset):
    betterMatrix['{0}, {1}'.format(pred, truth)] += 1

  # print better matrix
  print '\n\nBETTER MATRIX:\n'
  for key in betterMatrix.keys():
    val = betterMatrix[key]
    print '{0} => {1}'.format(key, val)

trainFeaturesFile.close()
trainResponseFile.close()
testFeaturesFile.close()
if testResponseFile:
  testResponseFile.close()
if predictedResponseFile:
  predictedResponseFile.close() 
