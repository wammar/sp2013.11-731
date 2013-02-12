#!/bin/bash

EVAL_FROM="100"

# compile
cd fsts
make -f Makefile-hmm
cd ../

# train forward hmm, and align the corpus
mpirun -np 7 ./fsts/train-hmm data/dev-test-train.de-en.src.600 data/dev-test-train.de-en.tgt.600 data/dev-test-train.de-en.hmm-fwd &> data/hmm-fwd.log & 
python extract-features.py -st data/dev-test-train.de-en -af data/dev-test-train.de-en.hmm-fwd.train.align -an hmm-fwd -gf data/dev.align -ff data/dev-test-train.de-en.hmm-fwd.features -rf data/dev.align.response -evalFrom $EVAL_FROM &> data/hmm-fwd.creg.log 

# train backward hmm
mpirun -np 7 ./fsts/train-hmm data/dev-test-train.de-en.tgt.600 data/dev-test-train.de-en.src.600 data/dev-test-train.de-en.hmm-bwd &> data/hmm-bwd.log &
python extract-features.py -st data/dev-test-train.de-en -af data/dev-test-train.de-en.hmm-bwd.train.align -an hmm-bwd -gf data/dev.align -ff data/dev-test-train.de-en.hmm-bwd.features -rf data/dev.align.response -bwd True -evalFrom $EVAL_FROM &> data/hmm-bwd.creg.log 

# train a classifier, and predict the alignment points, and convert to moses format
python train-and-test-scores.py -tf data/dev-test-train.de-en.hmm-fwd.features.train,data/dev-test-train.de-en.hmm-bwd.features.train -ef data/dev-test-train.de-en.hmm-fwd.features.eval,data/dev-test-train.de-en.hmm-bwd.features.eval -tr data/dev.align.response.train -er data/dev.align.response.eval -ao data/dev.align.pred &

# evaluate the classifier output
./check < data/dev.align.pred
./grade < data/dev.align.pred
