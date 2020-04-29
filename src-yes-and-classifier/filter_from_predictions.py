import json 
import re 
from argparse import ArgumentParser
from pathlib import Path 

def replace_bad_characters(text): 

    text = re.sub('\x82', ',', text)
    text = re.sub('\x84', ',,', text)
    text = re.sub('\x85', '...', text)
    text = re.sub('(\x97)+', ' - ', text)
    text = re.sub('(\x96)+', "-", text)
    text = re.sub('(\x95)|(\x99)|(\xa0)', " ", text)
    text = re.sub('(\x92)|(\x91)', "'", text)
    text = re.sub('(\x93)|(\x94)', '"', text)

    return text 

def filter(predictions, args): 
    ''' 
    Filter predictions based on criteria 
    ''' 
    valid_ending_punctuation = ['.', '!', '?']
    potential_yesands = [] 
    prompts = [] 
    repetitions = 0 
    for prediction in predictions: 
        # confidence must be above the threshold 
        if prediction['confidence']['yesand'] < args.threshold*100:
            continue 

        # process characters that MTURK cannot process: 
        prediction['p'] = replace_bad_characters(prediction['p'])
        prediction['r'] = replace_bad_characters(prediction['r'])

        if prediction['p'] in prompts: 
            repetitions += 1 
            continue 
        else: 
            prompts.append(prediction['p'])

        # remove instances with ... 
        if '...' in  prediction['p'] or '...' in prediction['r']: 
            continue 
        
        if prediction['p'][-1] not in valid_ending_punctuation or prediction['r'][-1] not in valid_ending_punctuation: 
            continue 

        # remove instances that are shorter than 4 words 
        if len(prediction['p'].split()) < 4 or len(prediction['r'].split()) < 4: 
            continue
        
        # if all criteria are passed, add to potential yes-ands 
        potential_yesands.append(prediction)

    print(f"There were {repetitions} in the prediction file")

    return potential_yesands


def main(): 
    parser = ArgumentParser() 
    parser.add_argument('--fp', type=str, help="File path of predictions to filter")
    parser.add_argument('--threshold',  type=float, default=0.95, help="Confidence threshold to filter classified results")

    args = parser.parse_args() 

    with open(args.fp, 'r') as f: 
        predictions = json.load(f) 

    potential_yesands = filter(predictions, args)

    proportion = round(len(potential_yesands) / len(predictions) * 100,2) 
    print(f"{len(potential_yesands)} predictions, {proportion}% of {len(predictions)} predictions, were yes-ands for a confidence threshold of {args.threshold}.")

    filtered_predictions_fp = Path(args.fp).parent / f'filtered_{args.threshold*100}_{Path(args.fp).name}'
    with open(filtered_predictions_fp, 'w') as f: 
        json.dump(potential_yesands, f, indent=4) 


if __name__ == "__main__": 
    main()