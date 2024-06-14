import os
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

from dataloader import return_data



def load_stocks(Data, keras_model):
    pred = keras_model.predict(Data)
    pred = pred.flatten()
    pred = (pred > 0.75).astype(int)
    cleaned = Data[pred == 1]
    return cleaned.index


def clean(current_date):
    # Load Encoder Models
    model_names = os.listdir('encoding_models')
    models = [load_model('encoding_models/' + name) for name in model_names if name != '.DS_Store']

    # Load Neural Network for predictions
    keras_model = load_model('predictive_models/fnnwa.h5')

    # Load Data Scaler
    scaler = joblib.load('predictive_models/data_scaler.joblib')

    # Get Data
    data = return_data()

    # Define Encoder Model Dict
    models_dict = {'RSI' : [models[4], ['rsi_30', 'rsi_60', 'rsi_120', 'rsi_240']], 
            'MACD' : [models[3],['macd_histogram_short', 'macd_histogram_medium', 'macd_histogram_long', 'macd_histogram_longest']], 
            'RVI' : [models[6], ['rvi_30', 'rvi_60', 'rvi_120', 'rvi_240']], 
            'ROC' : [models[0], ['roc_30', 'roc_60', 'roc_120', 'roc_240']], 
            'ATR' :[models[5], ['atr_30', 'atr_60', 'atr_120', 'atr_240']], 
            'MA' :[models[7], ['ma_30', 'ma_60', 'ma_120', 'ma_240']], 
            'VA' : [models[1], ['va_30', 'va_60', 'va_120', 'va_240']],
            'F' : [models[8], ['returnOnAssets', 'returnOnEquity', 'cashRatio', 'returnOnCapitalEmployed', 'equityRatio', 'netProfitMargin', 'OperatingProfitMargin', 'grossProfitMargin', 'debtRatio', 'accruals']],
            'FD' : [models[2], ['IncomeDelta', 'AssetsDelta', 'GrossMarginDelta']]}

    # Define Data that the model will see
    data['date'] = pd.to_datetime(data['date'])

    # Ensure the current_date is also in datetime format
    current_date = pd.to_datetime(current_date)
    # Filter the DataFrame based on the date and drop the 'date' column
    model_data = data[data['date'] == current_date].drop('date', axis=1)
    # Scale Data
    scaled_data = pd.DataFrame(
        scaler.transform(model_data),
        columns = model_data.columns
    )

    # Encode Data
    encoded_data = []
    for value in models_dict.values():
        encoded_data.append(value[0].predict(scaled_data[value[1]]).flatten())

    encoded_df = pd.DataFrame(data = np.array(encoded_data).T, columns=models_dict.keys())

    df = pd.concat([encoded_df, scaled_data], axis=1)

    df = df.set_index(data.index)

    # Stock Selection 
    stocks = load_stocks(df, keras_model)

    return stocks
