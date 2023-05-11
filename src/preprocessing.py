import torch
from scipy.io import loadmat as load
import numpy as np
from  numpy import pi as pi
from .OQPSK_Initialization import *
from .utils import TorchComplex as tc
from torch.nn.utils.rnn import pad_sequence

def freq_compensation_batch(
        segment, 
        freq, 
        PROCESS_SAMPLING_RATE=Process_Sampling_Rate
        ):
    N, segment_length = segment.shape[0], segment.shape[1]
    n = torch.arange(0, segment_length).view(1, -1, 1).to(segment.device).repeat(N, 1, 1)
    freq = freq.view(N, 1, 1).repeat(1, segment_length, 1)/PROCESS_SAMPLING_RATE*n*2*pi
    freq = torch.cat([freq*0, freq], dim=2) 
    exp_freq = tc.exp(freq)
    return tc.prod(segment, exp_freq)

def freq_compensation(
        segment, 
        freq, 
        PROCESS_SAMPLING_RATE=Process_Sampling_Rate
        ):
    segment_length = len(segment)
    n = torch.arange(0, segment_length).view(-1, 1).to(segment.device)
    freq = freq/PROCESS_SAMPLING_RATE*n*2*pi
    freq = torch.cat([freq*0, freq], dim=1) 
    exp_freq = tc.exp(freq)
    return tc.prod(segment, exp_freq)

def freq_offset_estimation(
        segment, 
        DP_NUM=Despreading_Number,
        OQPSK_SAMPLE_PER_SYM=OQPSK_Sample_per_Symbol,
        PROCESS_SAMPLING_RATE = Process_Sampling_Rate,
    ):
    # shape of segment:(N, 1280, 2)
    sym_length = OQPSK_SAMPLE_PER_SYM*DP_NUM
    unified_OQPSK = segment[:, sym_length*1:sym_length*4, :]
    conj_unified_OQPSK = tc.conj(segment[:, sym_length*4 :sym_length*7, :])
    differ_OQPSK = tc.prod(unified_OQPSK, conj_unified_OQPSK)
    differ_average = torch.mean(differ_OQPSK, dim=1)
    differ_angle = tc.phase(differ_average)

    freq_offset_DS = differ_angle/DP_NUM/OQPSK_SAMPLE_PER_SYM/3/2/pi
    freq_offset_DS = freq_offset_DS * PROCESS_SAMPLING_RATE
    return freq_offset_DS


def phase_compensation(
        segment,
        phase
    ):
    segment_length = len(segment)
    exp_phase = np.exp(-1j*phase)
    exp_phase = tc.complex2tensor(exp_phase).view(-1, 2).repeat(segment_length, 1)
    segment = tc.prod(segment, exp_phase)
    return segment


def demodulation(
        segment, 
        max_index, 
        DP_NUM=Despreading_Number,
        OQPSK_SAMPLE_PER_SYM=OQPSK_Sample_per_Symbol,
        OQPSK_SYM_LENGTH = OQPSK_Symbol_Length,
        OQPSK_2530_DETECT = OQPSK_2530_Detection,
    ):
    segment_length = len(segment)
    sym_length = DP_NUM * OQPSK_SAMPLE_PER_SYM
    OQPSK_2530_DETECT = tc.array2tensor(OQPSK_2530_DETECT)
    segment_OQPSK_count = int(np.floor(float(segment_length-max_index)/sym_length))
    segment_OQPSK = torch.zeros(segment_OQPSK_count, DP_NUM, 2)
    
    # print('Data_OQPSK_Count=', segment_OQPSK_count)
    # print('Despreading_Number=', DP_NUM)
    # print('OQPSK_Sample_per_Symbol=', OQPSK_SAMPLE_PER_SYM)

    k = max_index
    
    segment_temp = segment[k:]
    for i in range(segment_OQPSK_count):      
        for n in range(DP_NUM):
            sym_segment = segment_temp[i*sym_length:i*sym_length+OQPSK_SYM_LENGTH,:]
            corr = tc.prod(sym_segment, OQPSK_2530_DETECT[:len(sym_segment),n,:])
            segment_OQPSK[i, n, :] += torch.sum(corr, dim=0)

    segment_OQPSK = segment_OQPSK/DP_NUM/OQPSK_SAMPLE_PER_SYM
    ## Unify Despreading Signal
    unified_dict = {}
    unified_dict['segments'] = torch.zeros(segment_OQPSK_count, 2)
    unified_dict['symbols'] = torch.argmax(tc.abs(segment_OQPSK), dim=1)

    for n in range(segment_OQPSK_count):
        m = unified_dict['symbols'][n]
        unified_dict['segments'][n] = segment_OQPSK[n,m]
    return unified_dict

def GetDataSegment(signals, segments, index):
    slide_length = len(segments)
    if(slide_length > 0):
        assert index < slide_length, 'Required segment index is out of range'    
        segment_length = segments[index,1] - segments[index,0] + 1
        output_segment = tc.array2tensor(signals[segments[index,0]:segments[index,1]+1,0])      
        power = tc.power(output_segment)
        output_segment = output_segment / power

    else:
        output_segment = 0
        segment_length = 0
        print('No Data Found \n')
    return output_segment, segment_length, power

def synchronization_coarse(
        segment, 
        init_search_freq = 0,
        SYN_WIN_LENGTH=Synchronization_Window_Length,
        SYN_SYM_LENGTH=Synchronization_Symbol_Length,
        SYN_SYM_SAMPLE = Synchronization_Symbol_Sample[:,0],
        SEARCH_FREQ_STEP = Searching_Frequency_Step,
        SEARCH_FREQ_END = Searching_Frequency_End,
        PROCESS_SAMPLING_RATE = Process_Sampling_Rate,
        SYN_FINE_THRESD = Synchronization_Fine_Theoreshold,
        ):
    segment = segment.clone().detach()
    segment_length = len(segment)
    SYN_SYM_SAMPLE = tc.array2tensor(SYN_SYM_SAMPLE)
    raw_segment = segment
    process_segment = segment
    freq_offset = 0
    ## [A]. Coarse FO
    if(segment_length>(SYN_WIN_LENGTH + SYN_SYM_LENGTH)):
        # print('segment_length=', segment_length)
        search_freq = init_search_freq
        is_get_correct_freq_offset = False
        while(search_freq <= SEARCH_FREQ_END):
            # find FO by enumeration
            process_segment = freq_compensation(segment, search_freq)

            syn_results = torch.zeros(SYN_WIN_LENGTH, 2)
            for n in range(SYN_WIN_LENGTH):
                corr = tc.prod(process_segment[n:n+SYN_SYM_LENGTH], SYN_SYM_SAMPLE)
                syn_results[n,:] += torch.sum(corr, dim=0)

            syn_results = tc.abs(syn_results / SYN_SYM_LENGTH)
            max_index = torch.argmax(syn_results) 
            # if good enough, return
            if syn_results[max_index] >= SYN_FINE_THRESD:
                is_get_correct_freq_offset = True
                freq_offset = search_freq
                break
            else:
                process_segment = freq_compensation(segment, -search_freq)
                syn_results = torch.zeros(SYN_WIN_LENGTH, 2)
                for n in range(SYN_WIN_LENGTH):
                    corr = tc.prod(process_segment[n:n+SYN_SYM_LENGTH], SYN_SYM_SAMPLE)
                    syn_results[n,:] += torch.sum(corr, dim=0)

                syn_results = tc.abs(syn_results / SYN_SYM_LENGTH)
                max_index = torch.argmax(syn_results) 
                if syn_results[max_index] >= SYN_FINE_THRESD:
                    is_get_correct_freq_offset = True
                    freq_offset = -search_freq
                    break
                else:
                    if init_search_freq != 0:
                        search_freq = 0
                        init_search_freq = 0
                    else:
                        # update search frequency
                        search_freq += SEARCH_FREQ_STEP
        
    else:
        is_get_correct_freq_offset = False

    return  process_segment, max_index, freq_offset, is_get_correct_freq_offset

def synchronization_fine(        
        segment, 
        max_index, 
        SYN_FINE_THRESD = Synchronization_Fine_Theoreshold,
        DS_ITER_NUM = DS_Iteration_Number,
        PE_START = Process_Estimation_Start,
        PE_END = Process_Estimation_End,
        DP_NUM=Despreading_Number,
        OQPSK_SAMPLE_PER_SYM=OQPSK_Sample_per_Symbol,
        PROCESS_SAMPLING_RATE = Process_Sampling_Rate,
        OQPSK_2530_SHR_SYM_SAMPLE = OQPSK_2530_SHR_Symbol_Sample[:, 0]
    ):
    segment_origin = segment.clone().detach()
    segment = segment.clone().detach()
    OQPSK_2530_SHR_SYM_SAMPLE = tc.array2tensor(OQPSK_2530_SHR_SYM_SAMPLE)
    segment_length = len(segment)
    freq_offset = 0.0
    for m in range(DS_ITER_NUM):
        unified_dict = demodulation(segment, max_index)

        ## Fine Frequency Offset Estimation Use Despreading Signal
        unified_OQPSK = unified_dict['segments']
        segment_OQPSK_count = 8
    
        conj_unified_OQPSK = tc.conj(unified_OQPSK)[PE_START:segment_OQPSK_count-PE_END + 1]
        unified_OQPSK = unified_OQPSK[PE_START-1:segment_OQPSK_count-PE_END]
        differ_OQPSK = tc.prod(unified_OQPSK, conj_unified_OQPSK)
        differ_average = torch.mean(differ_OQPSK, dim=0)
        differ_angle = tc.phase(differ_average)

        freq_offset_DS = differ_angle/DP_NUM/OQPSK_SAMPLE_PER_SYM/2/pi
        freq_offset_DS *= PROCESS_SAMPLING_RATE
        segment = freq_compensation(segment_origin, freq_offset_DS)
        freq_offset += freq_offset_DS.item()

    return segment, freq_offset                          

def synchronization_fine2(
        segment, 
        max_index, 
        SYN_FINE_THRESD = Synchronization_Fine_Theoreshold,
        DS_ITER_NUM = DS_Iteration_Number,
        PE_START = Process_Estimation_Start,
        PE_END = Process_Estimation_End,
        DP_NUM=Despreading_Number,
        OQPSK_SAMPLE_PER_SYM=OQPSK_Sample_per_Symbol,
        PROCESS_SAMPLING_RATE = Process_Sampling_Rate,
        OQPSK_2530_SHR_SYM_SAMPLE = OQPSK_2530_SHR_Symbol_Sample[:, 0],
        OQPSK_SYM_LENGTH = OQPSK_Symbol_Length,
    ):
    segment = segment.clone().detach()
    sym_length = OQPSK_SAMPLE_PER_SYM*DP_NUM
    segment_temp = segment[max_index:].clone().detach()
    unified_OQPSK = segment_temp[sym_length*1:sym_length*4]
    conj_unified_OQPSK = tc.conj(segment_temp[sym_length*4:sym_length*7])
    differ_OQPSK = tc.prod(unified_OQPSK, conj_unified_OQPSK)
    differ_average = torch.mean(differ_OQPSK, dim=0)
    differ_angle = tc.phase_np(differ_average)

    freq_offset_DS = differ_angle/DP_NUM/OQPSK_SAMPLE_PER_SYM/3/2/pi
    freq_offset_DS *= PROCESS_SAMPLING_RATE
    segment = freq_compensation(segment, freq_offset_DS)
    freq_offset = freq_offset_DS.item()
    return segment, freq_offset

def synchronization_phase(
        segment, 
        OQPSK_2530_SHR_SYM_SAMPLE=OQPSK_2530_SHR_Symbol_Sample[:, 0]
    ):
    OQPSK_2530_SHR_SYM_SAMPLE = tc.array2tensor(OQPSK_2530_SHR_SYM_SAMPLE)
    preamble = segment[:1285]
    conj_OQPSK_2530_SHR_SYM = tc.conj(OQPSK_2530_SHR_SYM_SAMPLE)
    mean_corr = torch.mean(tc.prod(preamble, conj_OQPSK_2530_SHR_SYM), dim=0)
    phase = tc.phase_np(mean_corr)
    return phase_compensation(segment, phase), phase


def file2segment(filename, index):
    total_data =load(filename)
    signals = total_data['Brush_Data_Temp']
    segment_index = total_data['Count_Data_Length_Sides']
    [segment, length, power] = GetDataSegment(signals, segment_index, index)
    return segment, length, power

def synchronization_all(segment):
    process_segment, max_index, \
         freq_offset, is_get_correct_freq_offset = synchronization_coarse(segment)
    if is_get_correct_freq_offset:
        process_segment, _ = synchronization_fine(process_segment, max_index)
        ## Phase Compensation
        conj_OQPSK_2530_SHR_SYM = torch.conj(OQPSK_2530_SHR_SYM_SAMPLE)
        mean_corr = torch.mean(tc.prod(preamble, conj_OQPSK_2530_SHR_SYM), dim=0)
        phase = tc.phase_np(mean_corr)
    else:
        print('Not Zigbee Format OQPSK Data')
    return process_segment[max_index:]

def synchronization_all_test(segment):
    process_segment1, max_index, \
         freq_offset1, is_get_correct_freq_offset = synchronization_coarse(segment)
    if is_get_correct_freq_offset:
        process_segment2, freq_offset2 = synchronization_fine(process_segment1, max_index)
    else:
        print('Not Zigbee Format OQPSK Data')
    return  process_segment1[max_index:], process_segment2[max_index:], freq_offset1, freq_offset2

def synchronization_inverse(
    preamble,
    segment,
    idx,
    rec_freq_offset,
    ):
    N = len(preamble)
    segment[idx:idx+N] = preamble
    rec_segment = freq_compensation(segment, rec_freq_offset)
    snr = tc.SNR(rec_segment, segment)
    return rec_segment, snr


def main(device_ids, test_ids, flag='data'):
    
    if len(device_ids)> 1:
        device_flag = '{}-{}'.format(device_ids[0], device_ids[-1])
    else:
        device_flag = str(device_ids[0])
    test_flag = '-'.join([str(i) for i in test_ids])
    save_file_name = '{}_dv{}_id{}.pth'.format(flag, device_flag, test_flag)
    print(save_file_name)
    save_data = {}
    save_data['length'] = []
    save_data['y'] = []
    save_data['idx'] = []
    save_data['coarse_freq'] = []
    save_data['fine_freq'] = []
    save_data['phase'] = []
    save_data['x_origin'] = []
    save_data['x_fopo'] = []


    for device_id in device_ids:
        for test_id in test_ids:
            filename = '/workspace/DATASET/'\
            'ZigBee/Original/A_No_{}_19dBm_{}.mat'.format(device_id+1, test_id)
            print('Load file {}'.format(filename))
            total_data =load(filename)
            signals = total_data['Brush_Data_Temp']
            segment_index = total_data['Count_Data_Length_Sides']
            init_freq = 0
            for segment_id in range(len(segment_index)):
                [segment, length, power] = GetDataSegment(signals, segment_index, segment_id)
                process_segment, max_index, \
                    freq_offset1, is_get_correct_freq_offset = synchronization_coarse(segment, init_freq)
                init_freq = freq_offset1
                print('{}/{}/{}/{} Coarse: freq_offset= {}, max_index= {}'.format(
                    device_id, test_id, segment_id, len(save_data['y']), freq_offset1, max_index))
                if is_get_correct_freq_offset:
                    process_segment_fo, freq_offset2 = synchronization_fine2(process_segment.clone().detach(), max_index)
                    process_segment_fopo, phase_offset = synchronization_phase(process_segment_fo.clone().detach())
                    print('{}/{}/{}/{}  Fine:  freq_offset= {} phase_offset= {}\n'.format(
                    device_id, test_id, segment_id, len(save_data['y']), freq_offset2, phase_offset))
                    save_data['x_origin'].append(segment.view(-1, 2))
                    save_data['x_fopo'].append(process_segment_fopo.view(-1, 2))
                    save_data['idx'].append(max_index)
                    save_data['length'].append(len(process_segment))
                    save_data['coarse_freq'].append(freq_offset1)
                    save_data['fine_freq'].append(freq_offset2)
                    save_data['phase'].append(phase_offset)
                    save_data['y'].append(device_id)
                else:
                    print('Not Zigbee Format OQPSK Data')
                

    save_data['x'] = pad_sequence(save_data['x'], batch_first=True, padding_value=0)
    print('data length:', len(save_data['y']))
    torch.save(save_data, './Dataset/processed/{}'.format(save_file_name))

def test2():
    file_name = '/workspace/DATASET/ZigBee/Original/A_No_1_19dBm_1.mat'
    segment, length, power = file2segment(file_name, 8)
    segment_origin = segment
    segment_origin, max_index, \
         freq_offset1, is_get_correct_freq_offset = synchronization_coarse(segment_origin)
    segment_origin, freq_offset2 =  synchronization_fine2(segment_origin, max_index)
    rec_segment = synchronization_inverse(
        tc.AWGN(segment_origin[max_index: max_index+1285], 30), 
        segment_origin, 
        max_index, -freq_offset1-freq_offset2)

    print(tc.SNR(rec_segment, segment))

    segment_coarse, max_index_rec, \
         freq_offset1_rec, is_get_correct_freq_offset = synchronization_coarse(rec_segment)
    rec_segment, freq_offset2_rec =  synchronization_fine2(segment_coarse, max_index)

    print(freq_offset1 + freq_offset2)
    print(freq_offset1_rec + freq_offset2_rec)
    print(max_index)
    print(max_index_rec)
    print(tc.SNR(rec_segment, segment_origin))


def test():
    # testing
    # file_name = '/workspace/DATASET/ZigBee/Original/A_No_2_19dBm_1.mat'
    # segment, length, power = file2segment(file_name, 0)
    # segment_origin = segment
    segment_origin = tc.array2tensor(test_Symbol_Sample[:,0])
    segment_origin, max_index, \
         freq_offset1, is_get_correct_freq_offset = synchronization_coarse(segment_origin)
    segment_origin, freq_offset_origin =  synchronization_fine2(segment_origin, max_index)
    print(freq_offset_origin)
    # segment_origin, freq_offset_origin =  synchronization_fine2(segment_origin, max_index)
    # print(freq_offset_origin)
    # test 
    segment_origin = segment_origin[max_index:]
    print('Freq_offset estimation:', freq_offset_estimation(segment_origin.clone().view(1, -1, 2)))
    # result_dict = demodulation(segment_origin, 0)
    # print(result_dict['symbols'])
    set_freq_offset = 10360
    set_segment_offset = 5
    set_phase_offset = 0.0

    segment_noisy = segment_origin.clone()
    # segment_noisy = phase_compensation(segment_noisy, set_phase_offset) 
    if set_segment_offset > 0:
        segment_noisy[set_segment_offset:] = segment_origin[:-set_segment_offset]
        segment_noisy[0:set_segment_offset] *= 0
    segment_noisy = freq_compensation(segment_noisy, set_freq_offset) 
    # print('mse:', torch.sum((segment_noisy-segment_noisy2)**2))
    
    segment_coarse, max_index, \
         freq_offset1, is_get_correct_freq_offset = synchronization_coarse(segment_noisy)
    segment_coarse, freq_offset1 = synchronization_fine2(segment_noisy, max_index)
    print('max index:{} ({})'.format(max_index, set_segment_offset))
    print('coarse freq offset:', freq_offset1)

    segment_fine, freq_offset2 = synchronization_fine2(segment_coarse, max_index)
    print('fine freq offset:', freq_offset2)
    print('freq offset:{} ({})'.format(freq_offset1+freq_offset2, set_freq_offset))

    # print('mse:', torch.sum((segment_origin-segment_fine)**2))
    result_dict = demodulation(segment_fine, max_index)
    print(result_dict['symbols'])

    # segment_fine, freq_offset3 = synchronization_fine2(segment_fine, max_index)
    # print('fine freq offset:', freq_offset3)

    # segment_final, phase_offset = synchronization_phase(segment_fine)
    # print('phase offset:{} ({})'.format(phase_offset, set_phase_offset))

    # result_dict = demodulation(segment_fine, max_index)
    # print(result_dict['symbols'])

def test3():
    # file_name = '/workspace/DATASET/ZigBee/Original/A_No_2_19dBm_1.mat'
    # segment, length, power = file2segment(file_name, 8)
    # segment_origin = segment
    segment_origin = tc.array2tensor(test_Symbol_Sample[:,0])
    # segment_origin, max_index, \
    #      freq_offset1, is_get_correct_freq_offset = synchronization_coarse(segment_origin)
    # segment_origin, _ =  synchronization_fine2(segment_origin, max_index)
    # test 
    # segment_origin = segment_origin[max_index:]
    result_dict = demodulation(segment_origin, 0)
    print(result_dict['symbols'])
    set_freq_offset = 0
    set_segment_offset = 0
    set_phase_offset = 0.1*pi

    segment_noisy = segment_origin.clone()
    segment_noisy = phase_compensation(segment_noisy, set_phase_offset) 
    if set_segment_offset > 0:
        segment_noisy[set_segment_offset:] = segment_origin[:-set_segment_offset]
        segment_noisy[0:set_segment_offset] *= 0
    segment_noisy = freq_compensation(segment_noisy, set_freq_offset) 
    # print('mse:', torch.sum((segment_noisy-segment_noisy2)**2))
    
    segment_coarse, max_index, \
         freq_offset1, is_get_correct_freq_offset = synchronization_coarse(segment_noisy)
    print('max index:{} ({})'.format(max_index, set_segment_offset))
    print('coarse freq offset:', freq_offset1)

    segment_fine, freq_offset2 = synchronization_fine2(segment_coarse, max_index)
    print('fine freq offset:', freq_offset2)
    print('freq offset:{} ({})'.format(freq_offset1+freq_offset2, set_freq_offset))

    segment_final, phase_offset = synchronization_phase(segment_fine)
    print('phase offset:{} ({})'.format(phase_offset, set_phase_offset))
    # print('mse:', torch.sum((segment_origin-segment_fine)**2))
    result_dict = demodulation(segment_fine, max_index)
    print(result_dict['symbols'])

if __name__ == '__main__':
    device_ids_list = [
        range(45),
        # range(45,54),
    ]
    test_ids_list = [
        [1],
        # [5],
        # [1,2,3,4,5],
    ]
    for i in range(len(test_ids_list)):
        main(device_ids=device_ids_list[i], test_ids=test_ids_list[i], flag='ZigBee')
    