Homework 3
==========

Waleed Ammar & Manaal Faruqui
-----------------------------

Summary:
- two-stage decoding
1. first stage is a standard stack-based decoder which allows reordering, handles OOVs, ... etc.
2. second stage is a beam sampler which starts with the k-best complete hypotheses in the first stage, and resamples them by merging two phrases, splitting a phrase into two, swapping two phrases, or retranslating a source phrase.

In the stack-based decoder, a few things made a good improvement: filtering the stacks based on a heuristic probability of the hypothesis which takes into account the future cost**, a distortion cost** and brevity penalty.

In the beam sampler, we compute the difference in lm score while resampling in an efficient way. Also, we filter the beam based on the marginalized score of a complete hypothesis. 

Since the decoder is fast (especially with small stack sizes), we decoded the test set using several configurations and combined them based on the marginal translation score, which helped a little. 

