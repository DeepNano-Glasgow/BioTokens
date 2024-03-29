import utils
import config
import logging
import numpy as np


import torch
from torch.utils.data import DataLoader

from train import train, test, translate
from data_loader import MTDataset
from utils import measurements_tokenizer_load
from model import make_model, LabelSmoothing
from generator import Generator
from tqdm import tqdm


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run():
    utils.set_logger(config.log_path)

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_par = torch.nn.DataParallel(model)
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
    test(test_dataloader, model, criterion)


def check_opt():
    """check learning rate changes"""
    import numpy as np
    import matplotlib.pyplot as plt
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    opt = get_std_opt(model)
    # Three settings of the lrate hyperparameters.
    opts = [opt,
            NoamOpt(512, 1, 20000, None),
            NoamOpt(256, 1, 10000, None)]
    plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
    plt.legend(["512:10000", "512:20000", "256:10000"])
    plt.show()


def one_sentence_translate(sent, beam_search=True):
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    BOS = measurements_tokenizer_load().bos_id()  # 2
    EOS = measurements_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + measurements_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    translation = translate(batch_input, model, use_beam=beam_search)
    return translation


def translate_example():
    #sent = "933 0 882 1033"
    
    #1041 0 1005 1107
    
    # tgt: K E P	1048 0 1009 1106
    # tgt: Y H V  933 0 882 1033
    # tgt: Y H R D  974 0 921 1042
    
    gen = Generator()

    evalpath = './data/json/eval.txt'
    f = open(evalpath, 'r')
    lines = f.readlines()
    total = 0
    correct = 0
    diff = 0
    for i in tqdm(range(len(lines))):
        input_val = lines[i].split("\t")[1].split('\n')[0]
        expect_tgt = lines[i].split("\t")[0]
        sent = input_val
        output = one_sentence_translate(sent, beam_search=True)
        combine = []
        tokens_pc = ''
        for j in output:
            if j != ' ':
                combine.append(j)
        combine.append('<EOS>')
        #print(combine)
        tokens_pot, tokens_cap = gen.get(combine)
        if len(tokens_cap) != 0:
            tokens_cap = tokens_cap[:-1]
        for j in tokens_pot:
            tokens_pc = tokens_pc + str(j) +' '
        for j in tokens_cap:
            tokens_pc = tokens_pc + str(j) +' '
        tokens_pc = tokens_pc[:-1]
        output_val = tokens_pc + '\n'
        inputv = input_val.split(" ")
        outputv = output_val[:- 1].split(" ")
        if len(inputv) != len(outputv):
            print('input_val = ', input_val)
            print('expect_tgt = ', expect_tgt)
            print('output = ', output)
            print('output_val = ', output_val)
            total += len(inputv)
        else:
            for j in range(len(inputv)):
                total += 1
                if(inputv[j] == outputv[j]):
                    correct += 1
                else:
                    diff += abs(int(inputv[j]) - int(outputv[j]))
                    print('input_val = ', input_val)
                    print('expect_tgt = ', expect_tgt)
                    print('output = ', output)
                    print('output_val = ', output_val)
    print('accuracy = ', correct/total)
    print('average_difference = ', diff/(total-correct))



if __name__ == "__main__":
    import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    import warnings
    warnings.filterwarnings('ignore')
    run()
    translate_example()
