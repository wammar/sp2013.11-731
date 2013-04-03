Homework 3
==========

Waleed Ammar & Manaal Faruqui
-----------------------------

We implement a stack based decoder which allows reordering.
We use a bit vector that represents at any time step how many source phrases have been translated.

Our decoder  is basically divided into two parts:-

1. This part contruscts a translation given the input sentence, tm and lm. 

a) We allow distortions to take place and assign a distortion threshold and distortion penalty. 
b) We implement a future cost for translations which is the best cost possible for translating the untranslated phrases in the source.

2. The sencond part takes the input of the first part and performs edits on it.

a) We try to modify it using, swapping and merging of adjacent phrase pairs in the source sentence.
b) We also try to split and reltranslate a given phrase in the source.

All these opertions have parameters which we have tuned to obtain the best performance.
