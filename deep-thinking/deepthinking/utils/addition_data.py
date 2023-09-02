"""
@author: neel04
adapted from the Deep Thinking repo. New Version
"""
import torch
import random
import numpy as np
import multiprocessing
from torch.utils.data import Dataset, DataLoader

class bADDataset(Dataset):
  '''
  Generates a dataset of binary addition problems
  '''
  def __init__(self, mode, samples, seqlen):
    assert mode in ["train", "val", "test"]

    self.mode = mode
    self.samples = samples
    self.seqlen = seqlen
    self.tokens = ['0', '1', '', '+', '=', ',', 'not', 'val', '[MASK]']

    self.tok_dict = {token: index for index, token in enumerate(self.tokens)}
    self.inv_tok_dict = {index: token for index, token in enumerate(self.tokens)}

    self.pad_token = self.tok_dict[''] # 2
    self.masking_token = self.tok_dict['[MASK]']

    self.tokenizer = lambda x: [self.tok_dict[token] for token in x]

    if self.mode == "train": # bounds for the length of digits in EACH number @ different stages
        self.upper_b = 10
        self.lower_b = 1
    elif self.mode == "val":
        self.upper_b = 12
        self.lower_b = 10
    else:
        self.upper_b = 15
        self.lower_b = 12

  def __len__(self) -> int:
    return self.samples
  
  def stringify(self, x: list):
    return ''.join(str(i) for i in x)

  def generate_data(self, tokenizer, n_digits: int):
    '''
    Generates a binary addition problem with n_digits number of digits
    '''
    # generate two random numbers with n_digits number of digits
    num1 = random.randint(0, 2**n_digits)
    num2 = random.randint(0, 2**n_digits)

    # convert the numbers to binary and create the sum in binary. Don't pad.
    num1_bin = bin(num1)[2:]
    num2_bin = bin(num2)[2:]
    sum_bin = bin(num1 + num2)[2:]

    # pad the numbers and the sum with zeros on the left side to the maximum length
    max_len = max(len(num1_bin), len(num2_bin), len(sum_bin))
    num1_bin, num2_bin, sum_bin = num1_bin.zfill(max_len), num2_bin.zfill(max_len), sum_bin.zfill(max_len)

    # concatenate the numbers and the sum with the operators
    src_seq = f'{num1_bin}+{num2_bin}'

    return tokenizer(src_seq), tokenizer(sum_bin)

  def pad_sequence(self, seq: torch.Tensor, max_len: int, rndm_num:int):
    if self.mode == "train":
        # pads the sequence with random number of pad tokens on both sides
        return [self.pad_token] * rndm_num + seq + [self.pad_token] * (max_len - len(seq) - rndm_num)
    else:
        # this is the default padding for val and test
        return seq[:max_len] + [self.pad_token] * (max_len - len(seq))

  def decode(self, x: torch.Tensor):
    x = x.view(-1).tolist()

    #if len(set(x)) >= 3: # check to see if its inputs, because outputs is binary in [0, 1]
    return ''.join(self.inv_tok_dict[int(elem)] if elem in self.tok_dict.values() else '' for elem in x).strip()

  def __getitem__(self, idx: int):
    n_digits = np.random.randint(self.lower_b, self.upper_b+1)
    src_seq, tgt_seq = self.generate_data(self.tokenizer, n_digits=n_digits)

    maxlen = max(len(src_seq), len(tgt_seq))
    rndm_num = random.randint(0, self.seqlen - maxlen) if (self.seqlen - maxlen) >= 0 else 0  # set it to 0 to prevent error

    padded_src_seq = torch.Tensor(self.pad_sequence(src_seq, self.seqlen, rndm_num)).long()
    padded_tgt_seq = torch.Tensor(self.pad_sequence(tgt_seq, self.seqlen, rndm_num)).long()

    return padded_src_seq, padded_tgt_seq

def prepare_addition_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=False):
    # We ignore the train_data and test_data rather than removing for compatibility reasons
    
    train_dataset = bADDataset(mode='train', samples=50_000, seqlen=32)
    val_dataset = bADDataset(mode='val', samples=5_000, seqlen=32)
    test_dataset = bADDataset(mode='test', samples=5_000, seqlen=32)

    num_cores = multiprocessing.cpu_count()

    trainloader = DataLoader(train_dataset,
                             num_workers=num_cores,
                             batch_size=train_batch_size,
                             shuffle=shuffle,
                             drop_last=True,
                             pin_memory=True,
                             prefetch_factor=32)

    valloader = DataLoader(val_dataset,
                             num_workers=num_cores,
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=64)

    testloader = DataLoader(test_dataset,
                             num_workers=num_cores,
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=64)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
    print(f'\nAddition dataloaders have been succesfully created!')
