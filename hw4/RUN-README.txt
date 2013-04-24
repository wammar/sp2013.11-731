time python add_features.py data/dev.100best data/dev.100best.tokens.ru-en.align data/dev.src $1 | python pro.py --align data/dev.100best.tokens.ru-en.align --weights data/$1 --metric meteor --l1 100.0 --threshold 0.05
echo "Test:"
echo -n "BLEU: "
python add_features.py <(head -40000 data/test.100best) data/test.100best.tokens.ru-en.align data/test.src $1 | python rerank.py data/$1 | ./score-bleu -r data/test.ref
echo -n "METEOR: "
python add_features.py <(head -40000 data/test.100best) data/test.100best.tokens.ru-en.align data/test.src $1 | python rerank.py data/$1 | python score.py meteor data/test.ref
