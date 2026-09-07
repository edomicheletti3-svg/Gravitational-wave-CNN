import sys 
import numpy as np
from mly.tools import correlate 
from mly.datatools import DataSet

i = int(sys.argv[1])

training_set = DataSet.load(f'/home/edoardo.micheletti/model_tutorials/Models/training_data/training_subset_{i}_50k_14to32.pkl')
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
np.save(f'/home/edoardo.micheletti/model_tutorials/Models/C_grams/subset_{i}_COGRAM.npy',C_gram)