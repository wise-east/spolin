# Transform data to csv format
import json 
import pandas as pd 
from pathlib import Path 

def transform_spolin_data_to_rows(spolin_dict, split="train"): 
    rows = []
    ct = 0 
    for label, dict_ in spolin_dict.items(): 
        for source, pairs in dict_.items(): 
            for pair in pairs: 
                idx = f"{split}_{source}_{ct}"
                row ={
                    "idx": idx, 
                    "prompt": pair['p'],
                    "response": pair['r'],
                    "label": 1 if label=="yesands" else 0,
                    "source": source,
                    "split": split, 
                }
                rows.append(row)
                ct += 1 

    df = pd.DataFrame(rows)
    return df


files = ["data/spolin-train-acl.json", "data/spolin-train.json", "data/spolin-valid.json"]
for fn in files: 
    fn = Path(fn)
    csv_fn = fn.with_suffix(".csv")
    print(csv_fn)
    with fn.open("r") as f: 
        data = json.load(f) 

    split = "train" if "train" in str(fn) else "valid"
    df = transform_spolin_data_to_rows(data, split=split)

    df.to_csv(csv_fn, index=None)


    
