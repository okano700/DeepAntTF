import tensorflow as tf 
import numpy as np 
import pandas as pd
from tensorflow.python.autograph.operators.py_builtins import max_ 
from utils.TSds import TSds
from utils.WindowGen import WindowGenerator 
from utils.find_frequency import get_period
from utils.scorer import scorer
import argparse
from utils.DeepAnt import DeepAnt
import os

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
                      callbacks=[early_stopping, model_checkpoint_callback], verbose = 1)
    return history

if __name__ == "__main__":

    print(tf.config.list_physical_devices("GPU"))
    print(tf.test.is_built_with_cuda())

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type = str, required = True)
    parser.add_argument('--WL', type = int, required= True)
    parser.add_argument('--n', type= int, required= True)
    parser.add_argument('--i', type = int, required = True)
    parser.add_argument('--seed', type = int, required = True)
    parser.add_argument('--csv_name', type = str, required = True)

    args = parser.parse_args()

    print(args)

    SEQ_LEN = args.WL * args.n

    tf.random.set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    #Reading the data 
    ds = TSds.read_UCR(args.path)

    dataset = WindowGenerator(input_width = SEQ_LEN, label_width=1, shift = 1, train_df = ds.df[['value']].iloc[:ds.train_split], test_df= ds.df[["value"]].iloc[ds.train_split:], label_columns=['value'])

    #print(dataset)

    DA = DeepAnt(w_l = SEQ_LEN)

    hist = compile_and_fit(DA, dataset, patience = 10, MAX_EPOCHS=50)

    #converts values to TensorSliceDataset
    test_data = tf.data.Dataset.from_tensor_slices(dataset.scaler.transform(ds.ts.reshape(-1,1)) )
#takes window size  slices of the dataset
    test_data = test_data.window(SEQ_LEN, shift=1, drop_remainder=True)
#flattens windowed data by batching 
    test_data = test_data.flat_map(lambda x: x.batch(SEQ_LEN+1))
#creates batches of windows
    test_data = test_data.batch(32).prefetch(1)

    anomaly_score = DA.predict(test_data)
    anomaly_score = anomaly_score.reshape(-1,1)[-(len(ds.ts) - ds.train_split):] - dataset.scaler.transform(ds.ts[ds.train_split:].reshape(-1,1))
    #print(ds.df['is_anomaly'].iloc[ds.train_split:].shape, anomaly_score.squeeze().shape)
    _, _, res = scorer(ds.df['is_anomaly'].iloc[ds.train_split:].values, anomaly_score.squeeze())

    res['dataset'] = ds.name 
    res['WL'] = args.WL 
    res['n'] = args.n 
    res['id'] =args.i

    path_to_res = 'res_DATF_UCR.csv'
    
    if os.path.exists(path_to_res):
        df = pd.read_csv(path_to_res)
        pd.concat([df, pd.DataFrame([res])]).to_csv(path_to_res, index = False)

    else:
        pd.DataFrame([res]).to_csv(path_to_res, index = False)
    
