"""
@author: neel04
adapted from the Deep Thinking repo. New Version
"""

import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ArithmeticDataset(Dataset):
  def __init__(self, mode, samples, seqlen, bits):
    '''
    Setting up the dataset to produce an arithmetic datast
    with whatever operation
    '''
    assert mode in ['train', 'val', 'test']

    self.mode = mode
    self.samples = samples # Dataset size
    self.seqlen = seqlen
    self.bits = bits
    self.pad_token = 11

    if self.mode == 'train':
      self.upper_b = 6 #Bounding digits for different splits of the dataset
      self.lower_b = 1
    elif self.mode == 'val':
      self.upper_b = 7
      self.lower_b = 6
    else:
      self.upper_b = 7
      self.lower_b = 6

  def __len__(self):
    return self.samples
      
  def pad_text(self, text, length):
    if len(text) < length:
        return text + ' ' * (length - len(text))
    else:
        return text[:length]
  
  def ascii(self, x):
    if x == ' ':
      return self.pad_token 
    elif x == '+':
      return 12
    else:
      return ord(x) - 48 #shift to a narrower range

  def encode(self, text):
    padded_text = self.pad_text(text, self.seqlen)
    if '+' in text:
      ascii_con =  [self.ascii(char) for char in padded_text]
      #return torch.nn.functional.one_hot(torch.Tensor(ascii_con).long(), 13)
      return ascii_con
    else:
      return [self.ascii(char) for char in padded_text]
  
  def numpify(self, x):
    return np.array(list(x), dtype=np.int8)
  
  def rectangularity(self, x):
    return x.reshape(-1)

  def get_rndm_nums(self, N_low, N_high):
    # choose two digit lengths first
    N_low = 1 if N_low == 0 else N_low
    
    dig_1 = random.randint(N_low, N_high)
    dig_2 = random.randint(N_low, N_high)

    # Next, we generate the numbers themselves

    num_1 = random.randint(10**(dig_1-1), 10**dig_1 - 1)
    num_2 = random.randint(10**(dig_2-1), 10**dig_2 - 1)
    
    return num_1, num_2

  def decode(self, x):
    out = ''

    if len(x.shape) > 1:
      x = torch.argmax(x, dim=1)

    for elem in x:
      if elem != 12:
        out += chr(int(elem)+48)
      elif elem == 12:
        out += '+'
      else:
        out += ' '

    return out

  def __getitem__(self, idx):
    num1, num2 = self.get_rndm_nums(self.lower_b, self.upper_b)

    src_str = self.encode(f'{num1}+{num2}')
    tgt_str = self.encode(f'{num1+num2}')
    input, output = self.numpify(src_str), self.numpify(tgt_str)

    assert input.shape[0] == output.shape[0], f'shapes are wrong: {input.shape} and {output.shape}'
    return torch.from_numpy(input), torch.from_numpy(output)

def prepare_addition_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True):
    # We ignore the train_data and test_data rather than removing for compatibility reasons
    
    train_dataset = ArithmeticDataset(mode='train', samples=50_000, seqlen=16, bits=6)
    val_dataset = ArithmeticDataset(mode='val', samples=50_000, seqlen=16, bits=6)
    test_dataset = ArithmeticDataset(mode='test', samples=5_000, seqlen=16, bits=6)

    trainloader = DataLoader(train_dataset,
                             num_workers=2,
                             batch_size=train_batch_size,
                             shuffle=shuffle,
                             drop_last=True)

    valloader = DataLoader(val_dataset,
                             num_workers=2,
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=False)

    testloader = DataLoader(test_dataset,
                             num_workers=2,
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=False)
 
    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
    print(f'\nAddition dataloaders have been succesfully created!')
