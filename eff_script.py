import numpy as np
from mly.datatools import DataPod, DataSet, generator
import pickle as pkl
from mly.projectwave import *
from mly.exceptions import *
from mly.plugins import *
from mly.waveforms import *
from mly.tools import correlate 
import os  



def mass_pairs(total_min, total_max, step, m_floor=1.174):
    pairs = []
    for m1 in np.arange(m_floor,total_max,step):
        j_low = max(m1,total_min -m1)
        j_high = total_max - m1 
        for m2 in np.arange(j_low,j_high+step,step):
            if total_min <= m1 + m2 <= total_max and m2 >= m1:
                pairs.append((round(m1,5), round(m2,5)))
    return pairs
bns_waveform_list = []

mass = mass_pairs(2,3,step=0.01)
m = np.array(mass)
print("Total number of mass pairs generated: ", len(m))
m1 = m[:,0]
m2 = m[:,1]
fs = 1024


waveform_number = len(m1) 

set_size= 1000
for i in range(set_size):

    hp , hc = cbc(fs 
        ,m1[i]
        ,m2[i]
        ,spin1z=0
        ,spin2z=0
        ,aproximant='IMRPhenomT'
        ,inclination =0
        ,coalesence_phase = 0
        ,f_lower=60
        ,distance = 1
    )
    bns_waveform_list.append((hp,hc))

print("Waveform generation completed. Total waveforms generated: ", len(bns_waveform_list))

projected_bns_waveform_list = []
for bns in bns_waveform_list:
      b = projectWave(sourceWaveform=bns
                              ,detectors='HLV'
                              ,fs=fs
                              ,declination=None
                              ,rightAscension=None
                              ,polarisationAngle=None
                              ,time=0
                              ,outputFormat='datapod')
      projected_bns_waveform_list.append(b)


bns_injection_set = DataSet(projected_bns_waveform_list)
def coherencegram(strain, L=512, hop=128, maxlag=30):
    N = strain.shape[0]
    pairs = [(0,1),(0,2),(1,2)]                
    starts = list(range(0, N-L+1, hop))
    gram = np.zeros((len(starts), 2*maxlag, len(pairs)))
    for ti, s in enumerate(starts):
        for pi,(a,b) in enumerate(pairs):
            gram[ti,:,pi] = correlate(strain[s:s+L,a], strain[s:s+L,b],1024 ,maxlag)
    return gram  


os.environ["GWDATAFIND_SERVER"]="datafind.ldas.cit:80"

sources = [[1248705620.0, 1248709235.0]
           ,[1250647170, 1250650785]
           ,[1252364339, 1252367954]
           ,[1250650770, 1250654385]
           ,[1252367939, 1252371554]
           ,[1250654370, 1250657985]
           ,[1248705620, 1248709235]
           ,[1250657970, 1250661585]
           ,[1252376911, 1252380526]
           ,[1248709220, 1248712835]
           ,[1250661570, 1250665185]]

signal_set = {}
snrs = range(27,33)

index = range(0,len(sources)+1)
for snr,i in zip(snrs,index):
    signal_set[snr] = generator(duration=4,
                    fs=1024,
                    size=set_size,
                    labels={'type': 'signal'},
                    detectors='HLV',
                    backgroundType='real',
                    noiseSourceFile = [sources[i],sources[i],sources[i]],
                    windowSize=16,
                    injection_source=bns_injection_set,
                    injectionSNR=snr,
                    frames ={'H': 'H1_HOFT_C01', 'L': 'L1_HOFT_C01', 'V': 'V1Online'},
                    channels ={'H': 'H1:DCS-CALIB_STRAIN_C01', 'L': 'L1:DCS-CALIB_STRAIN_C01', 'V': 'V1:Hrec_hoft_16384Hz'},
                    whitening_config ={'method': 'welch', 'fftlength': 4, 'overlap': 2, 'f_min': 20, 'f_max': 512, 'processing_window': [10, 14]}
        )

C_gram = []
for snr in snrs:
    X = signal_set[snr].exportData('strain',shape = (None,4*1024,3))
    C_gram =  np.stack([coherencegram(x) for x in X])
    np.save(f'/home/edoardo.micheletti/model_tutorials/Models/C_grams/900-BNS-{snr}SNR-4seconds-COGRAM.npy',C_gram)
    print(f'Coherencegram with SNR {snr} has been saved')
    if snr != max(snrs):
        print('Job in Progress')
    elif snr == max(snrs): 
        print('Job done')