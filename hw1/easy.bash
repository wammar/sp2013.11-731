#!/bin/bash

EVAL_FROM="150"

# compile
cd fsts
#make -f Makefile-hmm
#make -f Makefile-model1
cd ../

# train model1 forward, and align the corpus
#mpirun -np 1 ./fsts/train-model1 data/dev-test-train.de-en.src data/dev-test-train.de-en.tgt data/dev-test-train.de-en.m1-fwd &> data/m1-fwd.log
python extract-features.py -st data/dev-test-train.de-en -af data/dev-test-train.de-en.m1-fwd.train.align -an m1-fwd -gf data/dev.align -ff data/dev-test-train.de-en.m1-fwd.features -rf data/dev.align.response -evalFrom $EVAL_FROM &> data/m1-fwd.creg.log

# train model1 backward, and align the corpus
#mpirun -np 1 ./fsts/train-model1 data/dev-test-train.de-en.tgt data/dev-test-train.de-en.src data/dev-test-train.de-en.m1-bwd &> data/m1-bwd.log
python extract-features.py -st data/dev-test-train.de-en -af data/dev-test-train.de-en.m1-bwd.train.align -an m1-bwd -gf data/dev.align -ff data/dev-test-train.de-en.m1-bwd.features -rf data/dev.align.response -bwd True -evalFrom $EVAL_FROM &> data/m1-bwd.creg.log

# train forward hmm, and align the corpus
#mpirun -np 14 ./fsts/train-hmm data/dev-test-train.de-en.src data/dev-test-train.de-en.tgt data/dev-test-train.de-en.hmm-fwd &> data/hmm-fwd.log  
#python extract-features.py -st data/dev-test-train.de-en -af data/dev-test-train.de-en.hmm-fwd.train.align -an hmm-fwd -gf data/dev.align -ff data/dev-test-train.de-en.hmm-fwd.features -rf data/dev.align.response -evalFrom $EVAL_FROM &> data/hmm-fwd.creg.log 

# train backward hmm
#mpirun -np 14 ./fsts/train-hmm data/dev-test-train.de-en.tgt data/dev-test-train.de-en.src data/dev-test-train.de-en.hmm-bwd &> data/hmm-bwd.log 
#python extract-features.py -st data/dev-test-train.de-en -af data/dev-test-train.de-en.hmm-bwd.train.align -an hmm-bwd -gf data/dev.align -ff data/dev-test-train.de-en.hmm-bwd.features -rf data/dev.align.response -bwd True -evalFrom $EVAL_FROM &> data/hmm-bwd.creg.log 

# train dice
#./align -n 100000 -t 0.8 > data/dev-test-train.de-en.dice.train.align
python extract-features.py -st data/dev-test-train.de-en -af data/dev-test-train.de-en.dice-fwd.train.align -an dice-bwd -gf data/dev.align -ff data/dev-test-train.de-en.dice-fwd.features -rf data/dev.align.response -evalFrom $EVAL_FROM &> data/dice-fwd.creg.log 

# train a classifier, and predict the alignment points, and convert to moses format
python train-and-test-scores.py \
    -tf data/dev-test-train.de-en.m1-fwd.features.train,data/dev-test-train.de-en.m1-bwd.features.train,data/dev-test-train.de-en.dice-fwd.features.train \
    -ef data/dev-test-train.de-en.m1-fwd.features.eval,data/dev-test-train.de-en.m1-bwd.features.eval,data/dev-test-train.de-en.dice-fwd.features.eval \
    -tr data/dev.align.response.train \
    -er data/dev.align.response.eval \
    -ao data/dev.align.pred 

# evaluate the classifier output
./check < data/dev.align.pred
./grade < data/dev.align.pred

# make sure it's fresh
ls -la data/dev.align.pred
