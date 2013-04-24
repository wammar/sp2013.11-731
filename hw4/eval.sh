#!/usr/bin/env bash
echo "Dev:"
echo -n "BLEU: "
python add_features.py data/dev.100best data/dev.100best.tokens.ru-en.align data/dev.src | python rerank.py $1 | ../score-bleu -r data/dev.ref
echo -n "METEOR: "
python add_features.py data/dev.100best data/dev.100best.tokens.ru-en.align data/dev.src | python rerank.py $1 | python score.py meteor data/dev.ref
echo "Test:"
echo -n "BLEU: "
python add_features.py <(head -40000 data/test.100best) data/test.100best.tokens.ru-en.align data/test.src | python rerank.py $1 | ../score-bleu -r data/test.ref
echo -n "METEOR: "
python add_features.py <(head -40000 data/test.100best) data/test.100best.tokens.ru-en.align data/test.src | python rerank.py $1 | python score.py meteor data/test.ref
