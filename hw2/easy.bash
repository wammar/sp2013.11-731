#!/bin/bash

# split the corpus
#python wammar-utils/split-corpus.py -r=1:1:1 -c=data/corpus.hyp1-hyp2-ref -t=data/train.hyp1-hyp2-ref -s=data/test.hyp1-hyp2-ref -d=data/dev.hyp1-hyp2-ref

# extract features
#python extract-features.py -i=data/dev.hyp1-hyp2-ref -f=data/dev.features
#python extract-features.py -i=data/train.hyp1-hyp2-ref -f=data/train.features

# train/test
#python train-and-test-scores.py -tf=data/train.features -tr=data/train.gold -sf=data/dev.features -sr=data/dev.gold -pr=data/dev.pred

# how well did we do?
#cat data/dev.pred | ./check | ./grade -i data/dev.hyp1-hyp2-ref -t data/dev.gold


# now, use the whole for training, and predict for the real-test
# extract features
python extract-features.py -i=data/corpus.hyp1-hyp2-ref -f=data/corpus.features
python extract-features.py -i=data/real-test.hyp1-hyp2-ref -f=data/real-test.features

# train/predict
python train-and-test-scores.py -tf=data/corpus.features -tr=data/corpus.gold -sf=data/real-test.features -pr=data/real-test.pred

# check for validity
cat data/real-test.pred | ./check > output.txt

