# SPOLIN

### SPOLIN: Spontaneanation Pairs of Learnable ImprovisatioN

This is the repo for the paper ["Grounding Conversations with Improvised Dialogues"]() (ACL2020). 
SPOLIN is a collection of "Yes, and" type dialogue pairs extracted from the Spontaneanation podcast by Paul F. Tompkins, the Cornell Movie Corpus, and the SubTle corpus.  

Core data is in the data/ folder. 


Project page: https://justin-cho.com/spolin

Demo: https://spolin.isi.edu




### Total 
|| Yes-ands| Non-yes-ands|
|--|---:|---:|
|Spontaneanation|10,959|6,087*|
|Cornell|16,926|18,810|
|SubTle|40,303|19,512|
|Total|68,188|44,409|

\*Artificially collected by mix & match of positive Spontaneanation samples


### Train set: `yesand_train.json`

|| Yes-ands| Non-yes-ands|
|--|---:|---:|
|Spontaneanation|10,459|5,587*|
|Cornell|16,426|18,310|
|SubTle|40,303|19,512|
|Total|67,188|43,409|


### Validation set: `yesand_valid.json`

|| Yes-ands| Non-yes-ands|
|--|---:|---:|
|Spontaneanation|500|500|
|Cornell|500|500|
|Total|1,000|1,000|

