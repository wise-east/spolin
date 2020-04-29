import numpy as np 
import torch
from utils import build_segment_ids, BERT_MAX_LEN
from keras.preprocessing.sequence import pad_sequences 
from transformers import BertTokenizer, BertForSequenceClassification
from argparse import ArgumentParser

def make_bert_input(prompt, response, tokenizer): 
    sentence = "[CLS] {} [SEP] {} [SEP]".format(prompt, response)
    input_ids = pad_sequences([tokenizer.encode(sentence)], maxlen=BERT_MAX_LEN, truncating="post", padding="post") 
    return input_ids

def predict_single(model, tokenizer, prompt, response): 
  
    # create inputs as batch size == 1
    input_ids = make_bert_input(prompt, response, tokenizer) 
    segment_ids = build_segment_ids(input_ids)
    attention_mask = []
    for seq in input_ids: 
        attention_mask.append([float(i>0) for i in seq])
    
    # expected shapes: (1, MAX_LEN), (1, MAX_LEN), (1_MAX_LEN)
    input_ids = torch.tensor(input_ids).long()
    segment_ids = torch.tensor(segment_ids).long()
    attention_mask = torch.tensor(attention_mask).long()
  
#   print(input_ids.shape, segment_ids.shape, attention_mask.shape)
  
    assert input_ids.shape == segment_ids.shape == attention_mask.shape
    
    model.eval()
    model.to('cpu')
    
    with torch.no_grad(): 
        logit = model(input_ids, segment_ids, attention_mask)
    
    confidence = torch.nn.functional.softmax(logit[0], dim=1)[0].numpy()
    label =  np.argmax(logit[0], axis=1).flatten()
    label = 'Yes, and' if label == 1 else 'Not a Yes, and'
  
    print("Prediction results:\n\tPrompt: {}\n\tResponse: {}\n\tLabel: {}\n\tConfidence: \n\t\tYes-and:     {:.2f}% \n\t\tNot Yes-and: {:.2f}%".format(prompt, response, label, confidence[1]*100, confidence[0]*100))
  
def main(): 
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", default="runs/yesand_bert_classifier8577", help="Provide a directory for a pretrained BERT model.")
    args = parser.parse_args()

    model = BertForSequenceClassification.from_pretrained(args.model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    while True: 
        prompt = input("Prompt: ")
        response = input("Response: ")

        predict_single(model, tokenizer, prompt, response)
        print('\n\n')


if __name__ == "__main__": 
    main() 