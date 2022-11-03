import torch
from scipy.io import loadmat as load
import numpy as np
from  numpy import pi as pi

is_debug = False
if is_debug:
    from OQPSK_Initialization import *
    from utils import TorchComplex as tc
    from utils import fft_plot
else:
    from .OQPSK_Initialization import *
    from .utils import TorchComplex as tc
    from .utils import fft_plot


OQPSK_2530_SHR_SYM_SAMPLE=OQPSK_2530_SHR_Symbol_Sample[:, 0]
OQPSK_2530_SHR_SYM_SAMPLE = tc.array2tensor(OQPSK_2530_SHR_SYM_SAMPLE)
conj_OQPSK_2530_SHR_SYM = tc.conj(OQPSK_2530_SHR_SYM_SAMPLE)


def freq_offset_estimation(
        segment, 
        DP_NUM=Despreading_Number,
        OQPSK_SAMPLE_PER_SYM=OQPSK_Sample_per_Symbol,
        PROCESS_SAMPLING_RATE = Process_Sampling_Rate,
    ):
    # shape of input segment:(N, T, 2)
    # return: (N,)
    sym_length = OQPSK_SAMPLE_PER_SYM*DP_NUM
    unified_OQPSK = segment[:, sym_length*1:sym_length*4, :]
    conj_unified_OQPSK = tc.conj(segment[:, sym_length*4 :sym_length*7, :])
    differ_OQPSK = tc.prod(unified_OQPSK, conj_unified_OQPSK)
    differ_average = torch.mean(differ_OQPSK, dim=1)
    differ_angle = tc.phase(differ_average)

    freq_offset_DS = differ_angle/DP_NUM/OQPSK_SAMPLE_PER_SYM/3/2/pi
    freq_offset_DS = freq_offset_DS * PROCESS_SAMPLING_RATE
    return freq_offset_DS

def freq_compensation(
        segment, 
        freq, 
        PROCESS_SAMPLING_RATE=Process_Sampling_Rate
        ):
    # shape of input segment:(N, T, 2)
    # shape of input freq:(N,)
    # return: (N, T, 2)
    N, segment_length = segment.shape[0], segment.shape[1]
    n = torch.arange(0, segment_length).view(1, -1, 1).to(segment.device).repeat(N, 1, 1)
    freq = freq.view(N, 1, 1).repeat(1, segment_length, 1)/PROCESS_SAMPLING_RATE*n*2*pi
    freq = torch.cat([freq*0, freq], dim=2) 
    exp_freq = tc.exp(freq)
    return tc.prod(segment, exp_freq)

def phase_offset_estimation(
        segment, 
        T = 1280,
        template=conj_OQPSK_2530_SHR_SYM
    ):
    # shape of input segment:(N, T, 2)
    # shape of input segment:(T, 2)
    # return: (N, T, 2)
    N = segment.shape[0]
    preamble = segment[:,:T,:]
    template = template.to(segment.device)
    template = template[:T, :].view(1, T, 2).repeat(N, 1, 1)
    mean_corr = torch.mean(tc.prod(preamble, template), dim=1)
    phase = tc.phase(mean_corr)
    return phase

def phase_compensation(
        segment,
        phase
    ):
    # shape of input segment:(N, T, 2)
    # shape of input phase:(N,)
    # return: (N, T, 2)
    N, T, _ = segment.shape
    neg_j_phase = torch.cat([phase.view(N, 1)*0, -1*phase.view(N, 1)], dim=1)
    exp_phase = tc.exp(neg_j_phase).view(N, 1, 2).repeat(1, T, 1)
    segment = tc.prod(segment, exp_phase)
    return segment

if __name__ == '__main__':
    temp0 = OQPSK_2530_SHR_SYM_SAMPLE.view(1, -1, 2).repeat(10, 1, 1)
    temp = freq_compensation(temp0, torch.Tensor([[1730]]).repeat(10, 1))
    freq = freq_offset_estimation(temp)
    print(freq)
    temp = freq_compensation(temp, freq)
    freq = freq_offset_estimation(temp)
    print(freq)
    temp = freq_compensation(temp, freq)
    freq = freq_offset_estimation(temp)
    print(freq)
    print(temp0-temp)
    fft_plot(temp[0], filename='test.png')
    fft_plot(temp0[0], filename='test0.png')


    temp = phase_compensation(temp, torch.Tensor([[3.14*0.4]]).repeat(10, 1))
    phase = phase_offset_estimation(temp)
    print(phase)
    temp = phase_compensation(temp, phase)
    phase = phase_offset_estimation(temp)
    # print(freq)
    print(phase)
