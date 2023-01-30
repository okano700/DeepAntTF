import numpy as np
import pandas as pd
import json

#def get_NABanomaly(dataset:str, pathRes:str, df: pd.DataFrame):
#    with open(pathRes) as jsonF:
#        anomaly = json.load(jsonF)
#    anomalies = np.zero(len(df))
#    for i in anomaly[dataset]:
#        anomalies[df.index.get_loc(i[0]):df.index.get_loc(i[1])]
    

def get_NABanomaly(df: pd.DataFrame, dfName:str, path:str):
    with open(path, "r") as jsonF:
        an = json.load(jsonF)
    aux = np.zeros(len(df))
    for start, end in an[dfName]:
        #print(df.index.get_loc(pd.to_datetime(start)), df.index.get_loc(pd.to_datetime(end)))
        aux[df.index.get_loc(pd.to_datetime(start)): df.index.get_loc(pd.to_datetime(end))] = 1
        #print(df.index.get_loc(pd.to_datetime(start))-df.index.get_loc(pd.to_datetime(end)))
    df['anomaly'] = aux  