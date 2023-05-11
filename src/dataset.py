import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from .utils import TorchComplex as tc
from .preprocessing import main as main_NMP
from .preprocessing_MP import main as main_MP

import os

def FIR(x, taps=9):
    _ , N, _ = x.shape
    padding = int((taps-1)/2)
    x_pad = torch.cat(
        [
            torch.zeros(padding, 2, dtype=x.dtype), 
            x.view(N, 2), 
            torch.zeros(padding, 2, dtype=x.dtype)
        ], dim=0)
    h = torch.randn(taps, 1, 2, dtype=x.dtype) / (taps*2)**0.5
    x_list = []
    for i in range(N):
        x_list.append(x_pad[i:i+taps, :])
    X = torch.stack(x_list)
    x_FIR = tc.mm(X, h)
    return x_FIR.view(1, N, 2)

class RFdataset(torch.utils.data.Dataset):
    def __init__(self, device_ids, test_ids, flag='ZigBee', SNR=None, rand_max_SNR=None, is_FIR=False):
        if len(device_ids)> 1:
            device_flag = '{}-{}'.format(device_ids[0], device_ids[-1])
        else:
            device_flag = str(device_ids[0])
        test_flag = '-'.join([str(i)  for i in test_ids])
        file_name = '{}_dv{}_id{}.pth'.format(flag, device_flag, test_flag)
        file_name = './datasets/processed/{}'.format(file_name)
        if not os.path.isfile(file_name):
            main_NMP(device_ids, test_ids, flag=flag)
     
        self.data = torch.load(file_name)

        self.snr = SNR
        self.max_snr = rand_max_SNR
        self.is_FIR = is_FIR

        
    def __getitem__(self, index):
        idx = self.data['idx'][index]      
        x_origin = self.data['x_origin'][index][idx:idx+1280, :].view(1, -1, 2).clone().detach()
        x_syn   = self.data['x_fopo'][index][idx:idx+1280, :].view(1, -1, 2).clone().detach()
        y        = self.data['y'][index]
        length   = self.data['length'][index]

        if self.is_FIR:
            x_origin = FIR(x_origin)
            
        if not self.snr is None:
            x_origin += tc.awgn(x_origin, self.snr, SNR_x=30)
            x_syn += tc.awgn(x_syn, self.snr, SNR_x=30)
            x += tc.awgn(x, self.snr, SNR_x=30)
        
        if not self.max_snr is None:
            rand_snr = torch.randint(5, self.max_snr, (1,)).item()
            x_origin += tc.awgn(x_origin, rand_snr, SNR_x=30)
            x_syn += tc.awgn(x_syn, rand_snr, SNR_x=30)
        
        return x_origin, y, x_syn

    def __len__(self):
        return len(self.data['y'])

class RFdataset_MP(torch.utils.data.Dataset):
    def __init__(self, device_ids, test_ids, flag='ZigBee', SNR=None, rand_max_SNR=None):
        if len(device_ids)> 1:
            device_flag = '{}-{}'.format(device_ids[0], device_ids[-1])
        else:
            device_flag = str(device_ids[0])
        test_flag = '-'.join([str(i)  for i in test_ids])
        file_name = '{}_dv{}_channel{}.pth'.format(flag, device_flag, test_flag)
        file_name = './datasets/processed/{}'.format(file_name)
        if not os.path.isfile(file_name):
            main_MP(device_ids, test_ids, flag=flag)
        self.data = torch.load(file_name)

        self.snr = SNR
        self.max_snr = rand_max_SNR

        
    def __getitem__(self, index):
        idx = self.data['idx'][index]
        x_origin = self.data['x_origin'][index][idx:idx+1280, :].view(1, -1, 2).clone().detach()
        x_syn   = self.data['x_fopo'][index][idx:idx+1280, :].view(1, -1, 2).clone().detach()
        y        = self.data['y'][index]
        length   = self.data['length'][index]

        if not self.snr is None:
            x_origin += tc.awgn(x_origin, self.snr, SNR_x=30)
            x_syn += tc.awgn(x_syn, self.snr, SNR_x=30)
        
        if not self.max_snr is None:
            rand_snr = torch.randint(5, self.max_snr, (1,)).item()
            x_origin += tc.awgn(x_origin, rand_snr, SNR_x=30)
            x_syn += tc.awgn(x_syn, rand_snr, SNR_x=30)
        return x_origin, y, x_syn

    def __len__(self):
        return len(self.data['y'])



if __name__ == "__main__":
    test = RFdataset_MP(device_ids=range(5), test_ids=[1,2,3], rand_max_SNR=None)
    print(len(test))
    print(test[0][0].shape)
    # min_freq = 1000000
    # max_freq = -1000000
    # sum_freq = 0.0
    # for i in range(len(test)):
    #     freq = test[i][0]
    #     if freq.max() > max_freq:
    #         max_freq = freq.max()
    #     if freq.min() < min_freq:
    #         min_freq = freq.min()
    #     sum_freq += freq.mean()
    # print(max_freq)
    # print(min_freq)
    # print(sum_freq/len(test))