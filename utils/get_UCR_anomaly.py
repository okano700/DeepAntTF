import pandas as pd
import numpy as np

def get_UCR_anomaly(df: pd.DataFrame, path:str):
    """
    get anomalies from the ucr time series anomaly
    -- parameters --
    df -> pd.DataFrame read from ucr anomaly
    path -> path of the file
    ----------------
    
    return 
    start, end, training
    """
    
    split_name = path.split('/')[-1]
    split_name = str(split_name).split('.')[0]
    name_aux = str(split_name).split('_')
    start = int(name_aux[5])
    end = int(name_aux[6])
    
    m_v = int(name_aux[4])
    
    aux = np.zeros(len(df))
    aux[start:end] = 1
    df['anomaly'] = aux

    return m_v