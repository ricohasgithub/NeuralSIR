
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from scipy.integrate import odeint
from sklearn.metrics import mean_absolute_error
import lmfit
from tqdm.auto import tqdm

import pickle
import joblib
import matplotlib.dates as mdates

# Prepare training data
df = pd.read_csv('data/data.csv', sep=';')

df.columns = ['date', 'region', 'total_infected', 'total_recovered', 'total_dead', 'deaths_per_day', 'infected_per_day', 'recovered_per_day']
df = df[df.region == 'Москва'].reset_index()
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df = df.drop(columns=['index', 'region'])
df = df.sort_values(by='date')
df.index = pd.date_range(start=df.date.iloc[0], end=df.date.iloc[-1], freq='D')

df_smoothed = df.rolling(7).mean().round(5)
df_smoothed.columns = [col + '_ma7' for col in df_smoothed.columns]

full_df = pd.concat([df, df_smoothed], axis=1)

for column in full_df.columns:
    if column.endswith('_ma7'):
        original_column = column.strip('_ma7')
        full_df[column] = full_df[column].fillna(full_df[original_column])

# Set training parameters for new variant modeling
NEW_STRAIN_DATE = '2021-01-10'

train_subset = df[
    #(df.date >= '2020-03-25') &
    (df.date <= NEW_STRAIN_DATE)]

# Class for classic SEIR forecasting model -- neural net to be inserted
# Model taken from: https://github.com/btseytlin/covid_peak_sir_modelling/blob/main/habr_code.ipynb
class SEIR:

    def __init__(self):
        self.params = self.get_fit_params()

    def get_fit_params(self):

        params = lmfit.Parameters()
        params.add("population", value=12_000_000, vary=False)
        params.add("epidemic_started_days_ago", value=10, vary=False)
        params.add("r0", value=4, min=3, max=5, vary=True)
        # CFR
        params.add("alpha", value=0.0064, min=0.005, max=0.0078, vary=True)
        # E -> I rate
        params.add("delta", value=1/3, min=1/14, max=1/2, vary=True)
        # I -> R rate
        params.add("gamma", value=1/9, min=1/14, max=1/7, vary=True)
        # I -> D rate
        params.add("rho", expr='gamma', vary=False)

        return params