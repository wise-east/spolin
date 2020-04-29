from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences 
from pathlib import Path 

import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification, AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers.optimization import WarmupLinearSchedule 
from argparse import ArgumentParser
from utils import build_segment_ids, BERT_MAX_LEN, ROBERTA_MAX_LEN, get_roberta_inputs 

import re 
import os
import json
import logging
from tqdm import tqdm 
from pprint import pformat

logger = logging.getLogger(__file__)
PREDICTION_BATCH_SIZE = 64


def get_data_loader(args, input_data, tokenizer):

    cache_fp = f"{args.data_path.replace('.json', '')}_{args.model_checkpoint}_cache"
    #forget about caching for now

    # to quickly test everything works, let input data be of one prediction batch size 
    if args.test:
        input_data = input_data[:PREDICTION_BATCH_SIZE]
    # format input as ROBERTA input 
    if 'roberta' in args.model: 
        input_ids, segment_ids, attention_masks = [], [], []
        for sample in input_data: 
            i, s, a = get_roberta_inputs(seq1=sample['p'], seq2=sample['r'], tokenizer=tokenizer)
            input_ids.append(i)
            segment_ids.append(s) 
            attention_masks.append(a) 

    # format sentence with BERT special tokens
    elif 'bert' in args.model: 
        sentences = []
        for sample in input_data: 
            sentence = '[CLS] {} [SEP] {} [SEP]'.format(sample['p'], sample['r'])
            sentences.append(sentence)

        # encode to BERT tokens
        logger.info("Tokenize input data...")
        input_ids = [tokenizer.encode(sentence) for sentence in sentences]

        # crop sequences longer than args.max_len
        for i in reversed(range(len(input_ids))): 
            if len(input_ids[i]) > args.max_len: 
                input_ids[i] = input_ids[i][:args.max_len]

        # pad to args.max_len
        input_ids = pad_sequences(input_ids, maxlen=args.max_len, truncating="post", padding="post")

        # get attention mask
        attention_masks = [] 
        for seq in input_ids: 
            attention_masks.append([float(i>0) for i in seq])

        # get segment information
        segment_ids = build_segment_ids(input_ids)


        

    # dialogue ids for tracking purposes 
    # need to match max_len 
    dialogue_idx = [[sample['idx']]*args.max_len for sample in input_data] 

    # wrap as tensors
    dialogue_idx = torch.tensor(dialogue_idx)
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    segment_ids = torch.tensor(segment_ids)

    # create dataloader
    prediction_data = TensorDataset(dialogue_idx, input_ids, attention_masks, segment_ids)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler = prediction_sampler, batch_size = PREDICTION_BATCH_SIZE)

    return prediction_dataloader

def predict_label(args, model, prediction_dataloader, data_to_predict): 
  
    # deactivate dropout, etc. 
    if torch.cuda.is_available(): 
        model.cuda()
        model.eval() 

    # for convenient retrieval 
    idx_to_data = {sample['idx']: {'p': sample['p'], 'r': sample['r']} for sample in data_to_predict}

    predictions = [] 
    for batch in tqdm(prediction_dataloader): 
    # add batch to GPU
        if torch.cuda.is_available(): 
            batch = tuple(t.to(args.device).to(torch.int64) for t in batch)

        #unpack input 
        b_dialogue_idx, b_input_ids, b_attention_masks, b_segment_ids = batch

        # don't store gradients
        with torch.no_grad(): 
            # forward pass
            if 'roberta' in args.model: 
                # logits = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_masks)
                logits = model(b_input_ids, token_type_ids=b_segment_ids, attention_mask=b_attention_masks)

            elif 'bert' in args.model: 
                logits = model(b_input_ids, b_attention_masks, b_segment_ids)
            

            softmax_logits = torch.nn.functional.softmax(logits[0], dim=1).cpu().numpy()

            # labels = softmax_logits[:,1] > args.threshold
            # labels = labels.astype(int).flatten()

            # TODO: modify how you want to store your prediction results
            for dialogue_idx, input_id, softmax_logit in zip(b_dialogue_idx, b_input_ids, softmax_logits): 
                idx = int(dialogue_idx[0].cpu().numpy())
                result = {'idx': idx, 'p': idx_to_data[idx]['p'], 'r': idx_to_data[idx]['r'], 'yesand-confidence':round(softmax_logit[1]*100, 2)}
                predictions.append(result)

    return predictions

def decode_input_id(input_id, tokenizer): 
    decoded_text = tokenizer.decode(input_id.to('cpu').numpy(), clean_up_tokenization_spaces=True)
    decoded_text_split = decoded_text[:2]
    decoded_text_split = [re.sub('(\[CLS\]|[-*+_])', '', split).strip() for split in decoded_text_split]
    decoded_text_split = [re.sub('\ \ ', ' ', split).strip() for split in decoded_text_split]

    return decoded_text_split[:2]

def get_list_data(data_path): 
    # For loading data already stored in list format
    # Each sample has the following format: {'id': int, 'p': str, 'r': str}

    with open(data_path, 'r') as f: 
        data_to_predict = json.load(f)

    return data_to_predict

def get_opus_data(data_path:str) -> dict: 
    # opus_data has the following format: {'filename': 'list of samples'}
    # Each sample in sublist has the following format: {'id': int, 'p': str, 'r': str}
    data_to_predict = []
    with open(data_path, 'r') as f: 
        data = json.load(f)

    idx = 0 
    for sublist in data.values(): 
        for item in sublist: 
            item.pop('id')
            item['idx'] = idx
            idx += 1 
        data_to_predict.extend(sublist)

    return data_to_predict 

def predict(): 
    """Determine which are yes-ands are not from a given dialogue data set with a finetuned BERT yes-and classifier"""
    parser = ArgumentParser()
    parser.add_argument("--data_path", help="Provide a datapath for which predictions will be made.")
    parser.add_argument("--predictions_folder", help="Provide a folderpath for which predictions will be saved to.")

    parser.add_argument("--model", default="bert-base-uncased", help="Provide pretrained model type that is consisten with BERT model that was fine-tuned.")
    parser.add_argument("--model_checkpoint", default="bert_postacl_with_selfyesand_v2", help="Provide a directory for a pretrained BERT model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--test", default=False, dest='test', action='store_true', help='runs validation after 1 training step')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: {}".format(pformat(args)))

    logger.info("Loading model and tokenizer.")
    if 'roberta' in args.model: 
        model = RobertaForSequenceClassification.from_pretrained(args.model_checkpoint)
        tokenizer = RobertaTokenizer.from_pretrained(args.model_checkpoint) 
        args.max_len = ROBERTA_MAX_LEN
    elif 'bert' in args.model: 
        model = BertForSequenceClassification.from_pretrained(args.model_checkpoint)
        tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint)
        args.max_len = BERT_MAX_LEN
    else: 
        error = f"Invalid model type given for args.model: {args.model}. Must either contain 'bert' or 'roberta"
        logger.info(error)
        return 

    logger.info("Loading data to predict: {}".format(args.data_path))

    if 'opus' in args.data_path: 
        data_to_predict = get_opus_data(args.data_path)
    else: 
        data_to_predict = get_list_data(args.data_path)

    logger.info("Building data loader...")
    prediction_dataloader = get_data_loader(args, data_to_predict, tokenizer)

    logger.info("Making predictions...")
    predictions = predict_label(args, model, prediction_dataloader, data_to_predict)
    logger.info("Predictions complete for {} dialogue pairs. ".format(len(predictions)))

    logger.info("Saving predictions...")

    if not Path(args.predictions_folder).is_dir(): 
        Path(args.predictions_folder).mkdir(parents=True, exist_ok=False)
    identifier = Path(args.data_path).name 
    checkpoint = Path(args.model_checkpoint).name 
    predictions_fp = f"{args.predictions_folder}pred_{checkpoint}_{identifier}"
    with open(predictions_fp, 'w') as f: 
        json.dump(predictions, f, indent=4)
    logger.info("Predictions saved to {}.".format(predictions_fp))

if __name__ == "__main__": 
    predict() 