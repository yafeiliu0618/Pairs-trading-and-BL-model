import pandas as pd
import numpy as np
def process_stock_data(input_csv):
    # Read the CSV file into a DataFrame
    stock_prices = pd.read_csv(input_csv)
    df = stock_prices.drop_duplicates()
    pivot_df = df.pivot(index='date', columns='PERMNO', values='PRC')
    pivot_df = pivot_df.dropna(axis=1)
    pivot_df = pivot_df.dropna(axis=1).abs()
    #permno_list = stock_prices['PERMNO'].tolist()
    return pivot_df

def calculate_percentage_change(pivot_df):
    df_ret = pivot_df.pct_change().dropna()
    return df_ret

def calculate_cumulative_returns(pivot_df):
    log_returns = np.log(pivot_df).diff()
    cumret = log_returns.cumsum() + 1
    cumret.dropna(inplace=True)
    return cumret