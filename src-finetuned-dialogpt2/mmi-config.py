import torch 

# bot token
TOKEN = 'put your bot token here'
# channel ids
talk_id = 647797094752714756
view_id = 648517874863833088

# device for forward and backward model
# device_f = 'cuda'
# device_r = 'cuda'

device_f = 'cuda' if torch.cuda.is_available() else 'cpu'
device_r = 'cuda' if torch.cuda.is_available() else 'cpu'


# sampling parameters
num_samples = 5
top_k = 20
MMI_temperature = 0.7

# other generation parameters 
focus_last_message = True