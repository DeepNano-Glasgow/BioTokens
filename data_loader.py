import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import measurements_tokenizer_load
from utils import amino_tokenizer_load

import config
DEVICE = config.device


def subsequent_mask(size):
   
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            trg = trg.to(DEVICE)
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.out_meas_sent, self.out_ami_sent = self.get_dataset(data_path, sort=True)
        self.sp_meas = measurements_tokenizer_load()
        self.sp_ami = amino_tokenizer_load()
        self.PAD = self.sp_meas.pad_id()  # 0
        self.BOS = self.sp_meas.bos_id()  # 2
        self.EOS = self.sp_meas.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        dataset = json.load(open(data_path, 'r'))
        out_meas_sent = []
        out_ami_sent = []
        for idx, _ in enumerate(dataset):
            out_meas_sent.append(dataset[idx][0])
            out_ami_sent.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(out_meas_sent)
            out_meas_sent = [out_meas_sent[i] for i in sorted_index]
            out_ami_sent = [out_ami_sent[i] for i in sorted_index]
        return out_meas_sent, out_ami_sent

    def __getitem__(self, idx):
        meas_text = self.out_meas_sent[idx]
        ami_text = self.out_ami_sent[idx]
        return [meas_text, ami_text]

    def __len__(self):
        return len(self.out_meas_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_meas.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_ami.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)
