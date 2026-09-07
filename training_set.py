import sys, numpy as np
from mly.datatools import DataSet, generator
from mly.waveforms import cbc, projectWave
import pickle as pkl


fs = 1024

set_size = 25000

def mass_pairs(total_min, total_max, step, m_floor=1.174):
    pairs = []
    for m1 in np.arange(m_floor,total_max,step):
        j_low = max(m1,total_min -m1)
        j_high = total_max - m1 
        for m2 in np.arange(j_low,j_high+step,step):
            if total_min <= m1 + m2 <= total_max and m2 >= m1:
                pairs.append((round(m1,5), round(m2,5)))
    return pairs

mass = mass_pairs(2,3, step = 0.002)

m = np.array(mass)
m1 = m[:,0]
m2 = m[:,1]
bns_waveform_list = []

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

signal_set = generator(duration=4,
                fs=1024,
                size=set_size,
                labels={'type': 'signal'},
                detectors='HLV',
                backgroundType='optimal',
                windowSize=16,
                plugins=['correlation_30'],
                injection_source=bns_injection_set,
                injectionSNR= [14,32]
)

noise_set = generator(duration=4,
                fs=1024,
                size= set_size,
                labels={'type': 'noise'},
                detectors='HLV',
                backgroundType='optimal',
                windowSize=16,
                plugins=['correlation_30']
                )

training_set = DataSet.fusion([signal_set,noise_set])
training_set.save(f'/home/edoardo.micheletti/model_tutorials/Models/training_data/50000_training_set_14to32.pkl')

X = training_set.exportData('strain', shape = (None,4*1024,3))

def coherencegram(strain, L=512, hop=128, maxlag=30):
    N = strain.shape[0]
    pairs = [(0,1),(0,2),(1,2)]                
    starts = list(range(0, N-L+1, hop))
    gram = np.zeros((len(starts), 2*maxlag, len(pairs)))
    for ti, s in enumerate(starts):
        for pi,(a,b) in enumerate(pairs):
            gram[ti,:,pi] = correlate(strain[s:s+L,a], strain[s:s+L,b],1024 ,maxlag)
    return gram  

C_gram = np.stack([coherencegram(x) for x in X])
np.save(f'/home/edoardo.micheletti/model_tutorials/Models/C_grams/50k_14_32_SNR_training_gram.npy',C_gram)
