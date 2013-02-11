import sys
import json
import argparse
import creg

sys.args = ['train-and-test-scores.py', '-tf', 'first10baselineFeatures.json', '-ef', 'last10baselineFeatures.json', '-tr', 'first10.json', '-er', 'last10.json']

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
        if int(obj['id']) != int(features['id']):
          print obj['id']
          print features['id']
      else:
        features[key] = obj[key]  
  if endOfFile:
    features = None
  return features;

def ReadResponse(originalFile):
  line = originalFile.readline()
  if(len(line) == 0):
    return None
  obj = json.loads(line)
  return int(obj[u'Score'])

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

  # pack it in creg.RealvaluedDataset
  trainData = creg.CategoricalDataset(trainData)
  return trainData

# parse arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-tf", "--trainFeaturesFilenames", type=str, help="comma-separated training features filenames")
argParser.add_argument("-ef", "--evalFeaturesFilenames", type=str, help="comma-separated evaluation features filenames")
argParser.add_argument("-tr", "--trainResponseFilename", type=str, help="comma-separated training response filename")
argParser.add_argument("-er", "--evalResponseFilename", type=str, help="comma-separated evaluation response filename")
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
trainDataset = CreateDataset(trainFeaturesFiles, trainResponseFile)

# train the model
model = creg.LogisticRegression(l2=1.0)
model.fit(trainDataset)
print 'model weights:'
print model.weights

# create eval dataset
evalDataset = CreateDataset(evalFeaturesFiles, evalResponseFile)

# evaluate
predictions = model.predict(evalDataset)
zeros, truth = [], []
for (x,y) in evalDataset:
  truth.append(y)
  zeros.append(0)
#print 'len(truth)={0}'.format(len(truth))
#print 'len(predictions)={0}'.format(len(predictions))
vs = zip(predictions, truth)
errors = sum((pred-real) ** 2 for (pred, real) in vs)
#print 'predictions vs. truth: \n{0}'.format(vs)
print 'mean squared error: %.3f' % (errors/float(len(evalDataset)))
