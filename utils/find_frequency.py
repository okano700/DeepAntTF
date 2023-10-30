from scipy.signal import periodogram
from math import floor
import numpy as np 
import argparse 
from utils.TSds import TSds


def get_period(data:np.array, n:int)-> list:
    f, px = periodogram(data, detrend='linear',nfft=int(len(data)*0.1) )
    p = []
    aux = 2
    for i in range(len(px)):
        #print(len(p))
        if len(p)>=n:
            break
        elif len(p) == 0:
            p.append(floor(1/f[np.argmax(px)] + 0.5))
        else:
            flag = False
            v = floor(1/f[px.argsort()[-aux]] + 0.5)
            for i in range(len(p)):
                
                if (p[i]%v != 0) and (v%p[i] != 0):
                    pass
                else:
                    flag = True
                    break
            if flag ==False:
                p.append(v)
            aux+=1
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type = str, required = True)
    parser.add_argument("--n", type = int, required = True)
    
    args = parser.parse_args()

    ds = TSds.read_UCR(args.path)

    print(get_period(ds.ts[:ds.train_split], args.n))

