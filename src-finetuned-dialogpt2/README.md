## Fine-tuned DialoGPT model with SPOLIN 

For fine-tuning the DialoGPT model with SPOLIN, follow the steps in the [DialoGPT](https://github.com/microsoft/DialoGPT) repository.  




#### Minimal Requirements for Inference

1. Make sure to [download]() and extract the model weights (`.pkl` file), `config.json`, `merges.txt`, and `vocab.json`. Also, download the [reverse GPT-2 model weights](https://convaisharables.blob.core.windows.net/lsp/multiref/small_reverse.pkl) and rename to `medium_reverse.pkl`.  All of these files must be in the same directory as `mmi-interact.py`. 

2. The minimal package requirements for inference, i.e. using `mmi-interact.py`, can be found in `requirements.txt`. Use this file to set up and activate the appropriate environment for using the scripts. 

3. Run `python3 mmi-interact.py`. 

You can change the decoding configurations in `mmi-config.py`. 


#### References

The MMI decoding method that is used for our [demo](https://spolin.isi.edu) is heavily based on [LHolten's implementation](https://github.com/LHolten/DialoGTP-MMI-decoder). 