# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:29:52 2019

@author: mystiacalbaby
"""
import numpy as np
from numpy import pi as pi

# Definition
global Process_Sampling_Rate
Process_Sampling_Rate = 10e6       # Definition of receiver sampling rate
OQPSK_Sample_per_Symbol = 10       # Setting samples per symbol at receiver
Despreading_Number = 16            # Spreading code numbers

# Definition for synchronization
Synchronization_Code = [0,0,0,0,0,0,0,0]  # Synchronization symbol is '0000'
Synchronization_Window_Length = 50         # Search max synchronization point with this window length
Do_Large_Scale_Frequency_Estimation = 1    # If frequency offset is very large, use this to get correct frequency offset

# Definition for symbol-scale frequency offset estimation
Process_Estimation_Start = 2         # Reduce some symbols at begining
Process_Estimation_End = 2            # Reduce some symbols at end
Searching_Frequency_Step = 1000 #1000        # When doing large scale frequency estimation, set this searching frequency step value
Searching_Frequency_End = 200000       # When doing large scale frequency estimation, set searching end
Synchronization_Fine_Theoreshold = 0.7 #0.7 # When doing large scale frequency estimation, set this theoreshold to get corret frequency estimation
Do_Re_Despreading = 1                  # Do despread again after frequency offset compensation
DS_Iteration_Number = 1 #1               # Set iteration number for frequency offset estimation with despreaded symbols
Do_Getting_Start_Phase = 1             # Do phase synchronization estimation

# Definition for get samples
OQPSK_Modulated_Symbol_Value = 1       # Definition of the symbol value when doing OQPSK modulation
Reduced_Symbol_Start = 14;              # Reduce some symbols at begining for geting one OQPSK symbol
Reduced_Symbol_End= 6                  # Reduce some symbols at end for geting one OQPSK symbol
Get_OQPSK_Sample_Window = 50           # For get start and end OQPSK symbol, set window size for averaging

# Initialization
Get_Data_Start = 0
Data_Search_End = 0
Get_Correct_Frequency_Offset = False
Find_Data_Indication = 0
Data_OQPSK_Count = 0

OQPSK_Symbol_Length = int(np.ceil(OQPSK_Sample_per_Symbol*Despreading_Number+OQPSK_Sample_per_Symbol/2))
OQPSK_2530 = np.zeros((OQPSK_Symbol_Length, Despreading_Number)) + np.zeros((OQPSK_Symbol_Length, Despreading_Number))*1j
Sine_Symbol = np.zeros((2*OQPSK_Sample_per_Symbol,1))

## Generate CC2530 OQPSK Standard Signal

data_map=[
'11011001110000110101001000101110',
'11101101100111000011010100100010',
'00101110110110011100001101010010',
'00100010111011011001110000110101',
'01010010001011101101100111000011',
'00110101001000101110110110011100',
'11000011010100100010111011011001',
'10011100001101010010001011101101',
'10001100100101100000011101111011',
'10111000110010010110000001110111',
'01111011100011001001011000000111',
'01110111101110001100100101100000',
'00000111011110111000110010010110',
'01100000011101111011100011001001',
'10010110000001110111101110001100',
'11001001011000000111011110111000']

for n in range(0,2*OQPSK_Sample_per_Symbol):
    Sine_Symbol[n,0] = np.sin(pi/OQPSK_Sample_per_Symbol*(n+1)-pi/OQPSK_Sample_per_Symbol/2)

for n in range(0,Despreading_Number):
    #Data=[data_map[n][0]]
    for m in range(0, Despreading_Number):
        Temp_1 = data_map[n][2*m]
        Temp_2 = int(Temp_1)
        if(Temp_2 == 1):
            for i in range(0,OQPSK_Sample_per_Symbol):
                OQPSK_2530[m*OQPSK_Sample_per_Symbol+i,n] = OQPSK_2530[m*OQPSK_Sample_per_Symbol+i,n] + Sine_Symbol[i,0]
        else:
             for i in range(0,OQPSK_Sample_per_Symbol):
                OQPSK_2530[m*OQPSK_Sample_per_Symbol+i,n] = OQPSK_2530[m*OQPSK_Sample_per_Symbol+i,n] + Sine_Symbol[i+OQPSK_Sample_per_Symbol,0]     
        
        Temp_1 = data_map[n][2*m+1]
        Temp_2 = int(Temp_1) 
        if(Temp_2 == 1):
            for i in range(0,OQPSK_Sample_per_Symbol):
                OQPSK_2530[int(OQPSK_Sample_per_Symbol/2)+m*OQPSK_Sample_per_Symbol+i,n] = OQPSK_2530[int(OQPSK_Sample_per_Symbol/2)+m*OQPSK_Sample_per_Symbol+i,n] + 1j*Sine_Symbol[i,0]
        else: 
             for i in range(0,OQPSK_Sample_per_Symbol):
                OQPSK_2530[int(OQPSK_Sample_per_Symbol/2)+m*OQPSK_Sample_per_Symbol+i,n] = OQPSK_2530[int(OQPSK_Sample_per_Symbol/2)+m*OQPSK_Sample_per_Symbol+i,n] + 1j*Sine_Symbol[i+OQPSK_Sample_per_Symbol,0]          

OQPSK_2530_Detection = np.conj(OQPSK_2530)



# Generate synchronization samples

Synchronization_Symbol_Length = len(Synchronization_Code)*OQPSK_Sample_per_Symbol*Despreading_Number + int(OQPSK_Sample_per_Symbol/2)
Synchronization_Symbol_Sample = np.zeros((Synchronization_Symbol_Length,1)) + 1j * np.zeros((Synchronization_Symbol_Length,1))

for n in range(0,len(Synchronization_Code)):
    Synchronization_Symbol_Sample[n*OQPSK_Sample_per_Symbol*Despreading_Number:(n+1)*OQPSK_Sample_per_Symbol*Despreading_Number,0] \
    = Synchronization_Symbol_Sample[n*OQPSK_Sample_per_Symbol*Despreading_Number:(n+1)*OQPSK_Sample_per_Symbol*Despreading_Number,0] \
    + np.real(OQPSK_2530[0:OQPSK_Sample_per_Symbol*Despreading_Number,Synchronization_Code[n]])
    
    Synchronization_Symbol_Sample[n*OQPSK_Sample_per_Symbol*Despreading_Number+int(OQPSK_Sample_per_Symbol/2): \
    (n+1)*OQPSK_Sample_per_Symbol*Despreading_Number+int(OQPSK_Sample_per_Symbol/2),0] =  \
    Synchronization_Symbol_Sample[n*OQPSK_Sample_per_Symbol*Despreading_Number+int(OQPSK_Sample_per_Symbol/2): \
    (n+1)*OQPSK_Sample_per_Symbol*Despreading_Number+int(OQPSK_Sample_per_Symbol/2),0] +  \
     1j*np.imag(OQPSK_2530[int(OQPSK_Sample_per_Symbol/2):OQPSK_Sample_per_Symbol*Despreading_Number \
                          +int(OQPSK_Sample_per_Symbol/2),Synchronization_Code[n]])

OQPSK_2530_SHR_Symbol_Sample = Synchronization_Symbol_Sample
#plt.plot(np.real(OQPSK_2530_SHR_Symbol_Sample))
Synchronization_Symbol_Sample = np.conj(Synchronization_Symbol_Sample)

# Generate test samples
test_Code = [0,0,0,0,0,0,0,0,7,10,10,2,3,4,5,0,0,0,0,0,0,0,0,0,0,0,0,15]
test_Symbol_Length = len(test_Code)*OQPSK_Sample_per_Symbol*Despreading_Number + int(OQPSK_Sample_per_Symbol/2)
test_Symbol_Sample = np.zeros((test_Symbol_Length,1)) + 1j * np.zeros((test_Symbol_Length,1))

for n in range(0,len(test_Code)):
    test_Symbol_Sample[n*OQPSK_Sample_per_Symbol*Despreading_Number:(n+1)*OQPSK_Sample_per_Symbol*Despreading_Number,0] \
    = test_Symbol_Sample[n*OQPSK_Sample_per_Symbol*Despreading_Number:(n+1)*OQPSK_Sample_per_Symbol*Despreading_Number,0] \
    + np.real(OQPSK_2530[0:OQPSK_Sample_per_Symbol*Despreading_Number,test_Code[n]])
    
    test_Symbol_Sample[n*OQPSK_Sample_per_Symbol*Despreading_Number+int(OQPSK_Sample_per_Symbol/2): \
    (n+1)*OQPSK_Sample_per_Symbol*Despreading_Number+int(OQPSK_Sample_per_Symbol/2),0] =  \
    test_Symbol_Sample[n*OQPSK_Sample_per_Symbol*Despreading_Number+int(OQPSK_Sample_per_Symbol/2): \
    (n+1)*OQPSK_Sample_per_Symbol*Despreading_Number+int(OQPSK_Sample_per_Symbol/2),0] +  \
     1j*np.imag(OQPSK_2530[int(OQPSK_Sample_per_Symbol/2):OQPSK_Sample_per_Symbol*Despreading_Number \
                          +int(OQPSK_Sample_per_Symbol/2),test_Code[n]])
