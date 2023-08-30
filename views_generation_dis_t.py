import pandas as pd
import os
import numpy as np
#from distance_method import calculate_distances
import statsmodels.api as sm
from scipy.stats import t
#from kalmanfilter import dynamic_regression

def calculate_distances(cumret):
    '''
    calculate Euclidean distance for each pair of stocks in the dataframe
    return sorted dictionary (in ascending order)
    '''
    import numpy as np
    distances = {} # dictionary with distance for each pair
    
    # calculate distances
    for s1 in cumret.columns:
        for s2 in cumret.columns:
            if s1!=s2 and (f'{s1}-{s2}'not in distances.keys()) and (f'{s2}-{s1}' not in distances.keys()):
                dist = np.sqrt(np.sum((cumret[s1] - cumret[s2])**2)) # Euclidean distance
                distances[f'{s1}-{s2}'] = dist
    
    # sort dictionary
    sorted_distances = {k:v for k,v in sorted(distances.items(), key = lambda item: item[1])}
    return sorted_distances

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



def calculate_results(cumret,df_ret,coint_start_day, coint_end_day, folder_path1, folder_path2,t0):
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
        output_file_path = os.path.join(folder_path2, f"pview_df_iteration_{iteration_count}.csv")

# Read the CSV file into a DataFrame
        pview_df = pd.read_csv(output_file_path)
        Opt_pairs = []
        for row_index, row in pview_df.iterrows():
            stock1 = row[row == 1].index[0]
            stock2 = row[row == -1].index[0]
            Opt_pairs.append((stock1, stock2))
        Opt_pairs = [(int(pair[0]), int(pair[1])) for pair in Opt_pairs]

        print(Opt_pairs)
        opt_pairs_df = pd.DataFrame(Opt_pairs, columns=['Stock 1', 'Stock 2'])
        print(opt_pairs_df)
      
        num_pairs = len(Opt_pairs)


        start_day = df_ret.index[end_day_index - t0] if end_day_index - t0 >= 0 else df_ret.index[0]

        if (end_day_index + rolling_days) < num:
            end_day = df_ret.index[end_day_index]
        else:
            end_day = df_ret.index[num-1]

        for i in range(num_pairs):
            stock_1 = df_ret.loc[start_day:end_day, Opt_pairs[i][0]]
            stock_2 = df_ret.loc[start_day:end_day, Opt_pairs[i][1]]
            #opt_pairs_df[Opt_pairs[i][0]] = stock_1  # Create a column for stock 1 and populate it with data
            #opt_pairs_df[Opt_pairs[i][1]] = stock_2  # Create a column for stock 2 and populate it with data
            #olsresults = sm.OLS(stock_1, stock_2).fit() 
            #predict = olsresults.predict(stock_2)
            #a = np.subtract(predict,stock_1)/stock_1
            diff_a = stock_1 - stock_2
            estimated_df, estimated_loc, estimated_scale = t.fit(diff_a)
            num_simulations = 252
            num_periods = 22
            t_distr = t(df=estimated_df, loc=estimated_loc, scale=estimated_scale)
            diff_random_numbers_t = t_distr.rvs(size=(num_simulations, num_periods))

            diff_simulated_spreads_t = np.zeros((num_simulations, num_periods))
            diff_simulated_spreads_t[:, 0] = diff_a[-1]

            for j in range(1, num_periods):
                diff_simulated_spreads_t[:, j] = diff_simulated_spreads_t[:, j - 1] + diff_random_numbers_t[:, j] 

# Calculate mean using t-distribution with estimated parameters
            diff_monte_carlo_mean_t = np.mean(np.mean(diff_simulated_spreads_t, axis=1)) * np.sqrt(22)
            print("diff Monte Carlo simulation return with t-distribution:", diff_monte_carlo_mean_t)

            diff_spread_mean = np.mean(diff_a)
            diff_spread_std = np.std(diff_a)
            num_simulations = 252
            num_periods = 22
            diff_random_numbers = np.random.normal(diff_spread_mean, 1, (num_simulations, num_periods))
            diff_simulated_spreads = np.zeros((num_simulations, num_periods))
            diff_simulated_spreads[:, 0] = diff_a[-1]
            for j in range(1, num_periods):
                diff_simulated_spreads[:, j] = diff_simulated_spreads[:, j-1] + diff_random_numbers[:, j] * diff_spread_std
            diff_monte_carlo_mean = np.mean(np.mean(diff_simulated_spreads, axis=1)) * np.sqrt(22)
            print("diff Monte Carlo simulation return modified:", diff_monte_carlo_mean)

            
            olsresults = sm.OLS(stock_1, stock_2).fit() 
            predict = olsresults.predict(stock_2)
            ols_a = predict - stock_2
            ols_estimated_df, ols_estimated_loc, ols_estimated_scale = t.fit(ols_a)
            ols_t_distr = t(df=ols_estimated_df, loc=ols_estimated_loc, scale=ols_estimated_scale)
            ols_random_numbers_t = ols_t_distr.rvs(size=(num_simulations, num_periods))

            ols_simulated_spreads_t = np.zeros((num_simulations, num_periods))
            ols_simulated_spreads_t[:, 0] = ols_a[-1]

            for j in range(1, num_periods):
                ols_simulated_spreads_t[:, j] = ols_simulated_spreads_t[:, j - 1] + ols_random_numbers_t[:, j] 

# Calculate mean using t-distribution with estimated parameters
            ols_monte_carlo_mean_t = np.mean(np.mean(ols_simulated_spreads_t, axis=1)) * np.sqrt(22)
            print("ols Monte Carlo simulation return with t-distribution:", ols_monte_carlo_mean_t)

            


            ols_spread_mean = np.mean(ols_a)
            ols_spread_std = np.std(ols_a)
            num_simulations = 252
            num_periods = 22
            ols_random_numbers = np.random.normal(ols_spread_mean, 1, (num_simulations, num_periods))
            ols_simulated_spreads = np.zeros((num_simulations, num_periods))
            ols_simulated_spreads[:, 0] = ols_a[-1]
            for j in range(1, num_periods):
                ols_simulated_spreads[:, j] = ols_simulated_spreads[:, j-1] + ols_random_numbers[:, j] * ols_spread_std
            ols_monte_carlo_mean = np.mean(np.mean(ols_simulated_spreads, axis=1)) * np.sqrt(22)
            print("ols Monte Carlo simulation return modified:", ols_monte_carlo_mean)
            
            kf_a = dynamic_regression(stock_1, stock_2, delta=1e-4)[2]
            kf_estimated_df, kf_estimated_loc, kf_estimated_scale = t.fit(kf_a)
            kf_t_distr = t(df=kf_estimated_df, loc=kf_estimated_loc, scale=kf_estimated_scale)
            kf_random_numbers_t = kf_t_distr.rvs(size=(num_simulations, num_periods))

            kf_simulated_spreads_t = np.zeros((num_simulations, num_periods))
            kf_simulated_spreads_t[:, 0] = kf_a[-1]

            for j in range(1, num_periods):
                kf_simulated_spreads_t[:, j] = kf_simulated_spreads_t[:, j - 1] + kf_random_numbers_t[:, j] 

# Calculate mean using t-distribution with estimated parameters
            kf_monte_carlo_mean_t = np.mean(np.mean(kf_simulated_spreads_t, axis=1)) * np.sqrt(22)
            print("kf Bootstrap Mean:", kf_monte_carlo_mean_t)
            
            kf_spread_mean = np.mean(kf_a)
            kf_spread_std = np.std(kf_a)
            num_simulations = 252
            num_periods = 22
            kf_random_numbers = np.random.normal(kf_spread_mean, 1, (num_simulations, num_periods))
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
                        'diff Monte Carlo Simulation t': diff_monte_carlo_mean_t,
                        'diff Monte Carlo Simulation modified': diff_monte_carlo_mean,
                        'ols Monte Carlo Simulation t': ols_monte_carlo_mean_t,
                        'ols Monte Carlo Simulation modified': ols_monte_carlo_mean,
                        'kf Monte Carlo Simulation t': kf_monte_carlo_mean_t,
                        'kf Monte Carlo Simulation modified': kf_monte_carlo_mean}
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