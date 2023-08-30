import pandas as pd
import glob

years = range(2000,2023) # Replace with the list of years you have
portfolio_coint_dfs = []
portfolio_distance_dfs = []
for year in years:
    portfolio_coint_path = f'/Users/liuyafei/Desktop/master thesis/{year}/portfolio_values_coint/*.csv'
    portfolio_distance_path = f'/Users/liuyafei/Desktop/master thesis/{year}/portfolio_values_distance/*.csv'
    
    portfolio_coint_files = glob.glob(portfolio_coint_path)
    portfolio_distance_files = glob.glob(portfolio_distance_path)
    
    
    for csv_file in portfolio_coint_files:
        df1 = pd.read_csv(csv_file)
        portfolio_coint_dfs.append(df1)
        
   
    for csv_file in portfolio_distance_files:
        df2 = pd.read_csv(csv_file)
        portfolio_distance_dfs.append(df2)
    
combined_portfolio_coint_df = pd.concat(portfolio_coint_dfs, ignore_index=True)
combined_portfolio_distance_df = pd.concat(portfolio_distance_dfs, ignore_index=True)