#!/bin/bash

# split the corpus
python wammar-utils/split-corpus.py -r=1:1:1 -c=data/corpus.hyp1-hyp2-ref -t=data/train.hyp1-hyp2-ref -s=data/test.hyp1-hyp2-ref -d=data/dev.hyp1-hyp2-ref

# extract features
python extract-features.py -i=data/dev.hyp1-hyp2-ref -f=data/dev.features
python extract-features.py -i=data/train.hyp1-hyp2-ref -f=data/train.features

# train/test
python train-and-test-scores.py -tf=data/train.features -tr=data/train.gold -sf=data/test.features -sr=data/test.gold -pr=data/test.pred

# how well did we do?
cat data/test.pred | ./check | ./evaluate -i=data/dev.hyp1-hyp2-ref -t=data/dev.gold

