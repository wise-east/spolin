import torch
import logging
import re
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from mmi_config import device_f, device_r, num_samples, MMI_temperature, top_k, focus_last_message
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Load and properly initialize models...")

torch.set_grad_enabled(False)

tokenizer = GPT2Tokenizer('vocab.json', 'merges.txt')

weights = torch.load('GP2-pretrain-step-615.pkl')
# weights = torch.load('models/medium/medium_ft.pkl')

# distributed training will prepend weights with 'module.'
keys = list(weights.keys())
for k in keys: 
    # if 'module.' in k: 
    weights[re.sub('module.', '', k)] = weights[k]
    weights.pop(k, None)

# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight", None)


cfg = GPT2Config.from_json_file('models/medium/config.json')
model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
model.load_state_dict(weights)
if device_f == 'cuda':
    model.half()
model.to(device_f)
model.eval()

logger.info("Loaded pretrained forward model")


weights = torch.load('medium_reverse.pkl')
# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight", None)

reverse_model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
reverse_model.load_state_dict(weights)
if device_r == 'cuda':
    reverse_model.half()
reverse_model.to(device_r)
reverse_model.eval()

logger.info("Loaded pretrained reverse model")


end_token = torch.tensor([[50256]], dtype=torch.long)


def _get_response(output_token, past):
    out = torch.tensor([[]], dtype=torch.long, device=device_f)

    while True:
        output_token, past = model.forward(output_token, past=past)
        output_token = output_token[:, -1, :].float()
        indices_to_remove = output_token < torch.topk(output_token, top_k)[0][..., -1, None]
        output_token[indices_to_remove] = -float('Inf')
        output_token = torch.multinomial(F.softmax(output_token, dim=-1), num_samples=1)

        if output_token.item() == end_token.item():
            break

        out = torch.cat((out, output_token), dim=1)

    return out, past


def _score_response(output_token, correct_token):
    inputs = torch.cat((output_token, correct_token), dim=1)
    mask = torch.full_like(output_token, -1, dtype=torch.long)
    labels = torch.cat((mask, correct_token), dim=1)

    loss, _, _ = reverse_model(inputs, labels=labels)

    return -loss.float()


def append_messages(old_list: list, new_list: list, truncate_length=128):
    for message in new_list:
        if message != '':
            input_token = tokenizer.encode(message, return_tensors='pt')
            input_token = torch.cat((input_token, end_token), dim=1)
            old_list.append(input_token)

    if len(old_list) == 0:
        old_list.append(end_token)

    # truncate
    total_length = 0
    for i, message in enumerate(reversed(old_list)):
        total_length += message.shape[1]

        ## very important 
        if total_length > truncate_length - 30:
            old_list[:] = old_list[-i:]
            logger.info("Truncating list.")
            print(old_list)


def generate_message(message_list: list, focus_last_message=True):
    total_input = torch.cat(message_list, dim=1).to(device_f)
    if focus_last_message:
        total_input_reversed = message_list[-1]
    else:
        total_input_reversed = torch.cat(list(reversed(message_list)), dim=1)


    past = None
    if total_input.shape[1] > 1:
        _, past = model(total_input[:, :-1])

    logger.info(total_input.shape)
    logger.info(total_input.tolist())
    logger.info("current input: " + tokenizer.decode(total_input.tolist()[0]))

    results = [_get_response(total_input[:, -1:], past) for i in range(num_samples)]
    # results = [_get_response(total_input[:, -1:], past) for i in range(num_samples)]

    results_with_scores = [result + (_score_response(result[0].to(device_r), total_input_reversed.to(device_r)), ) for result in results]

    return results_with_scores


if __name__ == '__main__':
    my_message_list = []
    while True:
        my_message = input('usr >> ')
        append_messages(my_message_list, [my_message])
        # my_response = generate_message(my_message_list)
        # print('bot >>', my_response)
        # append_messages(my_message_list, [my_response])

        start = time.time() 
        results = generate_message(my_message_list, focus_last_message)

        scores = F.softmax(torch.stack([x[2] for x in results], dim=0) / MMI_temperature, dim=0)
        responses = [tokenizer.decode(r[0].tolist()[0], skip_special_tokens=True) for r in results]

        for idx in range(len(results)): 
            print(f"\n\t{idx}: Score - {round(scores[idx].item(), 2)} Response - {responses[idx]}")

        end = time.time() 
        print(f"Time elapsed: {round(end - start, 2)}")

        choice = None 
        while choice not in range(len(results)): 
            try: 
                choice = int(input("Choose the index of the bot response to use: "))
            except Exception as e: 
                print(f"Error: {e}\nInvalid choice. Choose an index that is listed above.")
            
        append_messages(my_message_list, [responses[choice]])

