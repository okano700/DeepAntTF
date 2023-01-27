import numpy as np
import pandas as pd
import json
import urllib2

def get_NABanomaly(dataset:str, pathRes:str, df: pd.DataFrame):
    with open(pathRes) as jsonF:
        anomaly = json.load(jsonF)
    anomalies = np.zero(len(df))
    for i in anomaly[dataset]:
        anomalies[df.index.get_loc(i[0]):df.index.get_loc(i[1])]
    
        