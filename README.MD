# SPOLIN
[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

### SPOLIN: Spontaneanation Pairs of Learnable ImprovisatioN

This is the repo for the paper ["Grounding Conversations with Improvised Dialogues"]() (ACL2020). 
SPOLIN is a collection of more than 68,000 "Yes, and" type dialogue pairs extracted from the Spontaneanation podcast by Paul F. Tompkins, the Cornell Movie-Dialogs Corpus, and the SubTle corpus.  


#### Available SPOLIN versions:
The core dataset that was used for the experiments in the paper only includes _yesands_ and non-_yesands_ from Spontaneanation and most of what is provided in those extracted from the Cornell Movie-Dialogs Corpus. After the submitting the paper, we continued our iterative data augmentation process, repeating another iteration with the Cornell Movie-Dialogs Corpus and extracting from the SubTle corpus. This expanded version is also included in this repository [here](/data). This latest version of SPOLIN was used to train the model used in our demo. 


In the `data` folder, we provide two versions of the SPOLIN training set: 
1. Version used for experiments in the ACL paper: `data/spolin-train-acl.json` 
2. Expanded version: `data/spolin-train.json`


### Upcoming updates: 

We will release publicly available download links to our classifier and fine-tuned GPT-2 models soon. 

#### Relevant links: 

Project page: https://justin-cho.com/spolin
Demo: https://spolin.isi.edu
Paper: https://arxiv.org/abs/2004.09544



#### Latest SPOLIN:  
|| yesands| non-yesands|
|--|---:|---:|
|Spontaneanation|10,959|6,087*|
|Cornell|16,926|18,810|
|SubTle|40,303|19,512|
|Total|68,188|44,409|

\*Artificially collected by mix & matching positive Spontaneanation samples to balance dataset for training classifier


<table>
<tr><th> data/spolin-train.json</th>
<th> data/spolin-valid.json </th></tr>
<tr><td>

|| yesands| non-yesands|
|--|---:|---:|
|Spontaneanation|10,459|5,587*|
|Cornell|16,426|18,310|
|SubTle|40,303|19,512|
|Total|67,188|43,409|

</td><td>

|| yesands| non-yesands|
|--|---:|---:|
|Spontaneanation|500|500|
|Cornell|500|500|
|Total|1,000|1,000|

</td></tr> </table>


### Citation

If you use data or code in this repository, please cite our ACL2020 paper: 
```
@inproceedings{cho2020spolin,
    title={Grounding Conversations with Improvised Dialogues},
    author={Cho, Hyundong and May, Jonathan},
    booktitle ={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    year={2020}
}  
```


### License


This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg