Things I have tried:
 - harmonic mean of unigram naive-precision and naive-recall (helps)
 - tokenize (helps)
 - use precision, recall, and their mean as features to train classfier(s) (helps)
 - find alignments
 - approximate matching of words based on levenshtein distance (helps)
 - use "alignment jump" features (doesn't help)
 - sentence length penatly (helps)
 - use word embeddings to determine word similarity (doesn't help with a hard threshold)
 - add ngram precision and ngram recall as features (only recall helps!)
