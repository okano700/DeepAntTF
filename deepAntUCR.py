import numpy as np
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay, classification_report

from utils.get_UCR_anomaly import get_UCR_anomaly
from utils.get_period import get_period
from utils.WindowGen import WindowGenerator
from utils.DeepAnt import DeepAnt
import glob
import os

from scipy.signal import periodogram
from math import floor

import gc


SEED = 42


def compile_and_fit(model, window,MAX_EPOCHS = 30, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    checkpoint_filepath = 'tmp/checkpoint'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      batch_size=32,
                      callbacks=[early_stopping,model_checkpoint_callback], verbose =0, workers=8)
    return history



tf.keras.utils.set_random_seed(SEED)
srcUCR = "/home/emerson/data/UCR_anomaly_dataset/"
UCR = [p for p in glob.glob(f"{srcUCR}/*.txt")]
print("Running UCR DS", flush = True)
res = []
res_f = []
UCR.sort()
b_s = 32
for path in UCR:
    #Rodando todos os datasets:
    
    #extraindo dados do ds
    split_name = str(path).split('/')[-1]
    
    split_name = str(split_name).split('.')[0]
    ds_name = split_name
    split_name = str(split_name).split('_')
    #ds_name = f"{split_name[1]}_{split_name[0]}_{split_name[3]}"
    begin = 78
    if int(split_name[0]) < begin:
        continue
    
    print(f"{ds_name}", flush = True)
    
    #extraindo o TS
    df_1 = np.genfromtxt(path)
    df = pd.DataFrame(df_1,columns=['value'])
    m_v = get_UCR_anomaly(df, path)
    #df.plot(figsize = (15, 6), title = 'NOISEBIDMC1', legend = False);
    
    #extraindo periodo
    period = get_period(np.array(df['value'].loc[:m_v]), 3)
    
    #loop por periodo

    for p in period:
        print(f"Running period{p}", flush = True)
        #loop for mult
        for m in [2,3,5]:
            for it in range(5):

                #create model
                tf.keras.backend.clear_session()
                w_l = p*m
                p_w = 1

                ds = WindowGenerator(input_width= w_l, label_width= p_w, shift = 1, 
                                    train_df = df['value'].loc[:m_v].to_frame(), 
                                    test_df = df['value'][m_v:].to_frame(), 
                                    val_df = df['value'].loc[:m_v].to_frame())
                model_deepAnt = DeepAnt(w_l = w_l)

                try:

                    hist = compile_and_fit(model_deepAnt, ds, patience =5, MAX_EPOCHS = 20)
                    model_deepAnt.load_weights('tmp/checkpoint')
                except:
                    print(f"error --> {ds_name}", flush = True)


                #save model
                save_dir = f"models/{ds_name}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                model_deepAnt.save(f'{save_dir}/{ds_name}_{p}_{m}_{it}')

                #validate model
                allds = ds.make_ds_pred(data = df['value'].to_frame())

                yhat, loss = model_deepAnt.get_loss(allds)
                preds_losses = pd.Series(loss, index = np.arange(w_l,len(df)))
                
                
                
                #results AUC

                fpr, tpr, thresholds = roc_curve(df['anomaly'].iloc[m_v:], preds_losses.loc[m_v:])
                res_auc = auc(fpr,tpr)

                res.append({'Dataset':ds_name, 'Period': p, 'Iteration': it, 'Multiplier': m, 'AUC': res_auc})

                pd.DataFrame(res).to_csv(f'res/AUC_UCR_{begin}.csv', index = False)

                #F-score by threshold

                #aux = []

                for tr in [0.01, 0.02, 0.03,0.05, 0.07,0.1,0.2,0.3,0.5]:
                    resf = classification_report(df['anomaly'].iloc[m_v:],[1 if it> tr else 0 for it in preds_losses.loc[m_v:] ],output_dict= True, target_names = ['normal','anomaly'], zero_division = 0)
                    res_f.append({'Dataset':ds_name,
                                'Window_length':w_l,
                                'Multiplier': m,
                                'threshold': tr,
                                'Period': p,
                                'precision':resf['anomaly']['precision'],
                                'recall':resf['anomaly']['recall'],
                                'f1-score':resf['anomaly']['precision'],
                                'acc':resf['accuracy']})
                pd.DataFrame(res_f).to_csv(f'fscore_{begin}.csv', index = False)
                del ds
                del model_deepAnt
                del preds_losses
                del fpr, tpr, thresholds, res_auc, allds
                
                gc.collect()
            
        
    
    
    