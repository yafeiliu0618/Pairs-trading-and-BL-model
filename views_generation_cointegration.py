import pandas as pd
import os
import numpy as np
#from cointegration_method import find_cointegrated_pairs
import statsmodels.api as sm
#from kalmanfilter import dynamic_regression
import statsmodels.api as sm
from itertools import combinations
import numpy as np
from pykalman import KalmanFilter
def dynamic_regression(xdata,ydata,delta=1e-4):
    observation_matrix=np.vstack([xdata,np.ones(xdata.shape[0])]).T[:, np.newaxis]
    
    #Delta coefficient will determine the frequency of rebalancing estimates
    trans_cov = delta / (1 - delta) * np.eye(2)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=observation_matrix,
                  observation_covariance=1.0,
                  transition_covariance=trans_cov)

    state_means, state_covs = kf.filter(ydata)

    slope=state_means[:,0]
    intercept=state_means[:,1]

    return slope,intercept,ydata.values-slope*xdata.values-intercept


def find_cointegrated_pairs(dataframe, critial_value=0.05):
    n_assets = len(dataframe.columns)
    pairs = []

    for i, j in combinations(range(n_assets), 2):
        asset1 = dataframe.iloc[:, i]
        asset2 = dataframe.iloc[:, j]

        # Perform the cointegration test
        result = sm.tsa.coint(asset1, asset2)

        # Get the p-value from the test
        p_value = result[1]

        # If the p-value is less than the critical value, consider the pair cointegrated
        if p_value < critial_value:
            pairs.append((dataframe.columns[i], dataframe.columns[j], p_value))

    # Sort the pairs by p-value in ascending order
    sorted_pairs = sorted(pairs, key=lambda x: x[2])

    return sorted_pairs[:7]  # Return the top 7 pairs

def calculate_results(cumret,df_ret,coint_start_day, coint_end_day, folder_path1, folder_path2,t):
    # Your existing code here
    # ...
    if not os.path.exists(folder_path1):
        os.makedirs(folder_path1)
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)


    end_day_index = cumret.index.get_loc(coint_end_day)
    num = np.size(cumret.index)
    rolling_days = 22
    results_df = pd.DataFrame(columns=['Iteration'])
    opt_pairs_df = pd.DataFrame(columns=['Stock 1', 'Stock 2'])
    #ticker_count_reduced = model32[5]
    iteration_count = 1  # Initialize iteration count
    pview_df = pd.DataFrame(columns=cumret.columns)

    while end_day_index < num:
              #Opt_pairs = power_statistics(df_ret, coint_start_day, coint_end_day)
        train12_6 = cumret.loc[coint_start_day:coint_end_day]
        results = find_cointegrated_pairs(train12_6)
        Opt_pairs = [(t[0], t[1]) for t in results][:7]
        print(f"Iteration {iteration_count}:")
        print(Opt_pairs)
        opt_pairs_df = pd.DataFrame(Opt_pairs, columns=['Stock 1', 'Stock 2'])
        print(opt_pairs_df)

        num_pairs = len(Opt_pairs)
        pview_df = pd.DataFrame(0, index=range(num_pairs), columns=cumret.columns)
        # Iterate over the selected asset pairs
        for i, pair in enumerate(Opt_pairs):
            pview_df.loc[i, pair[0]] = 1  # Set +1 for the first asset in the pair
            pview_df.loc[i, pair[1]] = -1 
        print(pview_df)# Set -1 for the second asset in the pair

        #output_file = f"pview_df_iteration_{iteration_count}.csv"

        # Save the pview_df DataFrame to a CSV file with the generated output file name
        #pview_df.to_csv(output_file, index=False)
        output_file_pview = os.path.join(folder_path2, f"pview_df_iteration_{iteration_count}.csv")
        pview_df.to_csv(output_file_pview, index=False)

        start_day = df_ret.index[end_day_index - t] if end_day_index - t >= 0 else df_ret.index[0]

        if (end_day_index + rolling_days) < num:
            end_day = df_ret.index[end_day_index]
        else:
            end_day = df_ret.index[num-1]

        for i in range(num_pairs):
            stock_1 = df_ret.loc[start_day:end_day, Opt_pairs[i][0]]
            stock_2 = df_ret.loc[start_day:end_day, Opt_pairs[i][1]]
            opt_pairs_df[Opt_pairs[i][0]] = stock_1  # Create a column for stock 1 and populate it with data
            opt_pairs_df[Opt_pairs[i][1]] = stock_2  # Create a column for stock 2 and populate it with data
            #olsresults = sm.OLS(stock_1, stock_2).fit() 
            #predict = olsresults.predict(stock_2)
            #a = np.subtract(predict,stock_1)/stock_1
            
            diff_a = stock_1 - stock_2
            diff_bootstrap_samples = []
            for _ in range(len(diff_a)):
                diff_bootstrap_sample = np.random.choice(diff_a, size=252, replace=True)
                diff_bootstrap_samples.append(diff_bootstrap_sample)
                
            diff_bootstrap_means = np.mean(diff_bootstrap_samples, axis=1)
            diff_bootstrap_std = np.std(diff_bootstrap_samples, axis=1)
            diff_bootstrap_percentiles = np.percentile(diff_bootstrap_samples, [2.5, 97.5], axis=1)
            diff_bootstrap_mean = np.mean(diff_bootstrap_means)* np.sqrt(22)
            print("diff Bootstrap Mean:", diff_bootstrap_mean)
            
            diff_spread_mean = np.mean(diff_a)
            diff_spread_std = np.std(diff_a)
            num_simulations = 252
            num_periods = 22
            diff_random_numbers = np.random.normal(0, 1, (num_simulations, num_periods))
            diff_simulated_spreads = np.zeros((num_simulations, num_periods))
            diff_simulated_spreads[:, 0] = diff_a[-1]
            for j in range(1, num_periods):
                diff_simulated_spreads[:, j] = diff_simulated_spreads[:, j-1] + diff_random_numbers[:, j] * diff_spread_std
            diff_monte_carlo_mean = np.mean(np.mean(diff_simulated_spreads, axis=1)) * np.sqrt(22)
            print("diff Monte Carlo simulation return:", diff_monte_carlo_mean)
            
            olsresults = sm.OLS(stock_1, stock_2).fit() 
            predict = olsresults.predict(stock_2)
            ols_a = predict - stock_2
            ols_bootstrap_samples = []
            for _ in range(len(ols_a)):
                ols_bootstrap_sample = np.random.choice(ols_a, size=252, replace=True)
                ols_bootstrap_samples.append(ols_bootstrap_sample)
                
            ols_bootstrap_means = np.mean(ols_bootstrap_samples, axis=1)
            ols_bootstrap_std = np.std(ols_bootstrap_samples, axis=1)
            ols_bootstrap_percentiles = np.percentile(ols_bootstrap_samples, [2.5, 97.5], axis=1)
            ols_bootstrap_mean = np.mean(ols_bootstrap_means)* np.sqrt(22)
            print("ols Bootstrap Mean:", ols_bootstrap_mean)
            
            ols_spread_mean = np.mean(ols_a)
            ols_spread_std = np.std(ols_a)
            num_simulations = 252
            num_periods = 22
            ols_random_numbers = np.random.normal(0, 1, (num_simulations, num_periods))
            ols_simulated_spreads = np.zeros((num_simulations, num_periods))
            ols_simulated_spreads[:, 0] = ols_a[-1]
            for j in range(1, num_periods):
                ols_simulated_spreads[:, j] = ols_simulated_spreads[:, j-1] + ols_random_numbers[:, j] * ols_spread_std
            ols_monte_carlo_mean = np.mean(np.mean(ols_simulated_spreads, axis=1)) * np.sqrt(22)
            print("ols Monte Carlo simulation return:", ols_monte_carlo_mean)
            
            kf_a = dynamic_regression(stock_1, stock_2, delta=1e-4)[2]
            kf_bootstrap_samples = []
            for _ in range(len(kf_a)):
                kf_bootstrap_sample = np.random.choice(kf_a, size=252, replace=True)
                kf_bootstrap_samples.append(kf_bootstrap_sample)
                
            kf_bootstrap_means = np.mean(kf_bootstrap_samples, axis=1)
            kf_bootstrap_std = np.std(kf_bootstrap_samples, axis=1)
            kf_bootstrap_percentiles = np.percentile(kf_bootstrap_samples, [2.5, 97.5], axis=1)
            kf_bootstrap_mean = np.mean(kf_bootstrap_means)* np.sqrt(22)
            print("kf Bootstrap Mean:", kf_bootstrap_mean)
            
            kf_spread_mean = np.mean(kf_a)
            kf_spread_std = np.std(kf_a)
            num_simulations = 252
            num_periods = 22
            kf_random_numbers = np.random.normal(0, 1, (num_simulations, num_periods))
            kf_simulated_spreads = np.zeros((num_simulations, num_periods))
            kf_simulated_spreads[:, 0] = kf_a[-1]
            for j in range(1, num_periods):
                kf_simulated_spreads[:, j] = kf_simulated_spreads[:, j-1] + kf_random_numbers[:, j] * kf_spread_std
            kf_monte_carlo_mean = np.mean(np.mean(kf_simulated_spreads, axis=1)) * np.sqrt(22)
            print(" kf Monte Carlo simulation return:", kf_monte_carlo_mean)
            
            #coint_start_day_test = df_ret.index[df_ret.index.get_loc(coint_start_day) + rolling_days]
            #coint_end_day_test = df_ret.index[df_ret.index.get_loc(coint_end_day) + rolling_days]
            #stock_1_check = df_ret.loc[coint_start_day_test:coint_end_day_test, Opt_pairs[i][0]]
            #stock_1_check_returns = (stock_1_check[-1]-stock_1_check[0])/stock_1_check[0]
            #stock_1_monthly_test = (1+stock_1_check).prod()-1 
            #print(stock_1_monthly_test)

            
            #stock_2_check = df_ret.loc[coint_start_day_test:coint_end_day_test, Opt_pairs[i][1]]
            #stock_2_monthly_test = (1+stock_2_check).prod()-1 
            #stock_2_check_returns = (stock_2_check[-1]-stock_2_check[0])/stock_2_check[0]
            #actual_residual = (stock_1_check[-1]-stock_2_check[-1])/stock_2_check[-1]
            #actual_residual = stock_1_monthly_test-stock_2_monthly_test
            #print("Actual Residuals", actual_residual)
            
            new_data = {'Iteration': iteration_count,
                        'Stock 1': Opt_pairs[i][0],
                        'Stock 2': Opt_pairs[i][1],
                        'diff Bootstrap Mean': diff_bootstrap_mean,
                        'diff Monte Carlo Simulation': diff_monte_carlo_mean,
                        'ols Bootstrap Mean': ols_bootstrap_mean,
                        'ols Monte Carlo Simulation': ols_monte_carlo_mean,
                        'kf Bootstrap Mean': kf_bootstrap_mean,
                        'kf Monte Carlo Simulation': kf_monte_carlo_mean}
                        #'Actual Residuals': actual_residual}
            
            new_row_df = pd.DataFrame([new_data])

        # Concatenate the new DataFrame with the existing 'results_df'
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

            #results_df = results_df.append(new_data, ignore_index=True)
            print(results_df)

        # Save the view matrix to
        print(f"--- End of Iteration {iteration_count} ---")
        iteration_count += 1  # Increment iteration count

        coint_start_day_index = df_ret.index.get_loc(coint_start_day)+ rolling_days
        if coint_start_day_index >= len(df_ret.index):
            print("Error: coint_start_day index is out of range")
        # Handle the error in an appropriate way
            break



        coint_start_day = df_ret.index[df_ret.index.get_loc(coint_start_day) + rolling_days]
        coint_end_day_index = df_ret.index.get_loc(coint_end_day) +rolling_days
        if coint_end_day_index < num:
            coint_end_day = df_ret.index[coint_end_day_index]
            end_day_index = coint_end_day_index 
        else:
            end_day_index = coint_end_day_index
                        

    return results_df

def filtered_data(results_df, folder_path):
    iteration_numbers = results_df['Iteration'].unique()

    for iteration in iteration_numbers:
        # Filter the data for the current iteration
        filtered_data = results_df[results_df['Iteration'] == iteration]
        print(filtered_data)

        output_file_results = os.path.join(folder_path, f"output_{iteration}.csv")
        filtered_data.to_csv(output_file_results, index=False)
