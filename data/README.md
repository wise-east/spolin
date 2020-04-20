
#### SPOLIN is provided in JSON format
-----

**First level keys:**
* `yesands`
* `non-yesands`

**Second level keys:**
* `spont`: shorthand for Spontaneanation
* `cornell`: shorthand for Cornell Movie Dialogue Corpus 
* `subtle`: (not present in `spolin-valid.json`) shorthand for  SubTle Corpus 

Each second level item is a list of _yesands_ or non-_yesands_. Each item contains the following: 
* `id`: dataset ID
* `prompt`: Utterance that starts the dialogue turn 
* `response`: Response to the prompt


##### `spolin-train.json`:  
|| yesands| non-yesands|
|--|---:|---:|
|Spontaneanation|10,459|5,587*|
|Cornell|16,426|18,310|
|SubTle|40,303|19,512|
|Total|67,188|43,409|


##### `spolin-train-acl.json`: 

|| yesands| non-yesands|
|--|---:|---:|
|Spontaneanation|10,459|5,587*|
|Cornell|14,976|17,851|
|Total|25,435|23,438|

##### `spolin-valid.json`: 

|| yesands| non-yesands|
|--|---:|---:|
|Spontaneanation|500|500*|
|Cornell|500|500|
|Total|1,000|1,000|

\*Artificially collected by mix & matching positive Spontaneanation samples to balance dataset for training classifier


