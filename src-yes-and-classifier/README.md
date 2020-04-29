# Fine-tune BertForSequenceClassification to train a sequence classifier

A pretrained BERT by [huggingface](https://github.com/huggingface/pytorch-transformers) can be easily used to train a sequence classifier using BERT. This repo uses a pretrained BERT train a 'yes, and' classifier, to determine whether a given dialogue pair is a ["Yes, and..."](https://en.wikipedia.org/wiki/Yes,_and...). 

## Requirements

Requirements can be found in `requirements.txt`.

## Finetuning steps

In practice, this code can be adjusted with minimal effort to train any downstream task that takes two sentences as input, such as entailment(NLI), etc. The only modifications required are the `get_data` and `build_bert_input` in `utils.py` to appropriately format the input data. They are marked with `#TODO` in the code. 

Once code for reformatting data is modified: 
1. Run `python train.py`. Check `train()` for default training parameters. 
2. Model checkpoints will be saved in `runs/`. To make predictions for a held-out dataset, run `python predict.py --model_checkpoint runs/<checkpoint directory> --data_path <datapath to held-out data>`. Check `predict()` for default parameters. 


### References

 - Code was based on a very useful [blog post](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) by Chris McCormick. Most of the code directly comes from his code while the code was refactored with `pytorch-ignite` and adjusted to incorporate the migration changes from `pytorch_pretrained_bert` to `pytorch_transformers`. The migration notes can be found [here](https://huggingface.co/pytorch-transformers/migration.html).

 - Use of `pytorch-ignite` to refactor some of the code was referenced from [huggingface's ConvAI chatbot implementation](https://github.com/huggingface/transfer-learning-conv-ai).
