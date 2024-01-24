import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 1517
tgt_vocab_size = 27
batch_size = 32
epoch_num = 30
early_stop = 5
lr = 3e-4

max_len = 10
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True

# data_dir = './data'
# train_data_path = './data/json/mytrain.json'
# dev_data_path = './data/json/mydev.json'
# test_data_path = './data/json/mytest.json'
# model_path = './experiment/model.pth'
# log_path = './experiment/train.log'
# output_path = './experiment/output.txt'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')

print(device)
