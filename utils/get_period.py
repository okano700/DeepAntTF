from scipy.signal import periodogram
from math import floor
import numpy as np
def get_period(data:np.array, n:int)-> list:
    f, px = periodogram(data, detrend='linear',nfft=int(len(data)*0.25) )
    
    return [floor(1/f[a] + 0.5) for a in px.argsort()[-n:][::-1]]