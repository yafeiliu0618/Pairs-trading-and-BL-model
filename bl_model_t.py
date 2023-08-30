import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
warnings.filterwarnings('ignore')

def black_litterman_portfolio(coint_start_day, coint_end_day, df_ret,folder_path1,folder_path2,t1,t2,t3,t4,t5,t6):
    end_day_index = df_ret.index.get_loc(coint_end_day)
    num = np.size(df_ret.index)
    rolling_days = 22
    iteration_count = 1  # Initialize iteration count
    portfolio_values1 = []
    portfolio_values2 = []
    portfolio_values3 = []
    portfolio_values4 = []
    portfolio_values5 = []
    portfolio_values6 = []
    results_df = pd.DataFrame(columns=['Iteration'])
    tau = 0.05

    while end_day_index < num:
        start_day = df_ret.index[end_day_index - t1] if end_day_index - t1>= 0 else df_ret.index[0]
        
        if (end_day_index + rolling_days) < num:
            end_day = df_ret.index[end_day_index]
        else:
            end_day = df_ret.index[num-1]
        ret_subset = df_ret.loc[start_day:end_day]
        #ret_subset = df_ret_full.loc[coint_start_day:coint_end_day]
        
        # Calculate the covariance matrix for the current iteration
        cov_matrix = ret_subset.cov()
        csv_file_path1 = os.path.join(folder_path1, f"output_{iteration_count}.csv")
        
        Q1 = pd.read_csv(csv_file_path1)["diff Monte Carlo Simulation t"]

        csv_file_path2 = os.path.join(folder_path2, f"pview_df_iteration_{iteration_count}.csv")
        P = pd.read_csv(csv_file_path2)
        #print(P)
        omega = tau * np.dot(np.dot(P,cov_matrix), P.T)
        #print(omega)
        bl_model1 = BlackLittermanModel(cov_matrix,Q=Q1,P=P,omega=omega)
        #bl_model1.bl_weights()
        rets1 = bl_model1.bl_returns()
        ef1 = EfficientFrontier(rets1,cov_matrix,weight_bounds=(-1, 1))
        #print(ef.min_volatility())
        bl1 = ef1.max_sharpe(risk_free_rate = -1)  
        #bl1 = bl_model1.clean_weights()
        bl1_df = pd.DataFrame.from_dict(bl1, orient='index', columns=['Weight'])

        #print(bl1_df)
        total_weight1 = sum(bl1.values())
        normalized_weights1 = pd.DataFrame((key, value/total_weight1) for key, value in bl1.items())
        #print(normalized_weights)
    #df_ret2.iloc[44,:]*normalized_weights.T
        #print(f"--- End of Iteration {iteration_count} ---")
        #iteration_count += 1  # Increment iteration count
        
        
        
        
        # Save the view matrix to
        coint_end_day_index = df_ret.index.get_loc(coint_end_day)
        if coint_end_day_index+rolling_days >= len(df_ret.index):
            print("Error: coint_start_day index is out of range")
        # Handle the error in an appropriate way
            break
        
    
        
        #rebalance_day = df_ret2.index[coint_end_day_index + rolling_days]
        rebalance_day = df_ret.index[coint_end_day_index + rolling_days] 
        #rebalance_asset = df_ret2.loc[rebalance_day]
        rebalance_asset = df_ret.loc[coint_end_day:rebalance_day]
        rebalance_asset_return = (1+rebalance_asset).prod()-1
        portfolio_value1= np.dot(np.array(rebalance_asset_return.T), np.array(normalized_weights1.iloc[:, 1]))
        portfolio_values1.append(portfolio_value1.tolist())
        #print(portfolio_value1)
        
        start_day = df_ret.index[end_day_index - t2] if end_day_index - t2>= 0 else df_ret.index[0]
        
        if (end_day_index + rolling_days) < num:
            end_day = df_ret.index[end_day_index]
        else:
            end_day = df_ret.index[num-1]
        ret_subset = df_ret.loc[start_day:end_day]
        #ret_subset = df_ret_full.loc[coint_start_day:coint_end_day]
        
        # Calculate the covariance matrix for the current iteration
        cov_matrix = ret_subset.cov()
        csv_file_path1 = os.path.join(folder_path1, f"output_{iteration_count}.csv")
        Q2 = pd.read_csv(csv_file_path1)["diff Monte Carlo Simulation modified"]

        csv_file_path2 = os.path.join(folder_path2, f"pview_df_iteration_{iteration_count}.csv")
        P = pd.read_csv(csv_file_path2)
        #print(P)
        omega = tau * np.dot(np.dot(P,cov_matrix), P.T)
        #print(omega)
        bl_model2 = BlackLittermanModel(cov_matrix,Q=Q2,P=P,omega=omega)
        rets2 = bl_model2.bl_returns()
        ef2 = EfficientFrontier(rets2,cov_matrix,weight_bounds=(-1, 1))
        #print(ef.min_volatility())
        bl2 = ef2.max_sharpe(risk_free_rate = -1)  
        #bl_model2.bl_weights()  
        #bl2 = bl_model2.clean_weights()
        bl2_df = pd.DataFrame.from_dict(bl2, orient='index', columns=['Weight'])

        #print(bl2_df)
        total_weight2 = sum(bl2.values())
        normalized_weights2 = pd.DataFrame((key, value/total_weight2) for key, value in bl2.items())
        #print(normalized_weights)
    #df_ret2.iloc[44,:]*normalized_weights.T
        #print(f"--- End of Iteration {iteration_count} ---")
        #iteration_count += 1  # Increment iteration count
        
        
        
        
        # Save the view matrix to
        coint_end_day_index = df_ret.index.get_loc(coint_end_day)
        if coint_end_day_index+rolling_days >= len(df_ret.index):
            print("Error: coint_start_day index is out of range")
        # Handle the error in an appropriate way
            break
        
    
        
        #rebalance_day = df_ret2.index[coint_end_day_index + rolling_days]
        rebalance_day = df_ret.index[coint_end_day_index + rolling_days] 
        #rebalance_asset = df_ret2.loc[rebalance_day]
        rebalance_asset = df_ret.loc[coint_end_day:rebalance_day]
        rebalance_asset_return = (1+rebalance_asset).prod()-1
        portfolio_value2= np.dot(np.array(rebalance_asset_return.T), np.array(normalized_weights2.iloc[:, 1]))
        portfolio_values2.append(portfolio_value2.tolist())
        #print(portfolio_value2)

        start_day = df_ret.index[end_day_index - t3] if end_day_index - t3>= 0 else df_ret.index[0]
        
        if (end_day_index + rolling_days) < num:
            end_day = df_ret.index[end_day_index]
        else:
            end_day = df_ret.index[num-1]
        ret_subset = df_ret.loc[start_day:end_day]
        #ret_subset = df_ret_full.loc[coint_start_day:coint_end_day]
        
        # Calculate the covariance matrix for the current iteration
        cov_matrix = ret_subset.cov()
        csv_file_path1 = os.path.join(folder_path1, f"output_{iteration_count}.csv")

        Q3 = pd.read_csv(csv_file_path1)["ols Monte Carlo Simulation t"]

        csv_file_path2 = os.path.join(folder_path2, f"pview_df_iteration_{iteration_count}.csv")
        P = pd.read_csv(csv_file_path2)
        #print(P)
        omega = tau * np.dot(np.dot(P,cov_matrix), P.T)
        #print(omega)
        bl_model3 = BlackLittermanModel(cov_matrix,Q=Q3,P=P,omega=omega)
        rets3 = bl_model3.bl_returns()
        ef3 = EfficientFrontier(rets3,cov_matrix,weight_bounds=(-1, 1))
        #print(ef.min_volatility())
        bl3 = ef3.max_sharpe(risk_free_rate = -1)  
        #bl_model3.bl_weights()  
        #bl3 = bl_model3.clean_weights()
        bl3_df = pd.DataFrame.from_dict(bl3, orient='index', columns=['Weight'])

        #print(bl3_df)
        total_weight3 = sum(bl3.values())
        normalized_weights3 = pd.DataFrame((key, value/total_weight3) for key, value in bl3.items())
        #print(normalized_weights)
    #df_ret2.iloc[44,:]*normalized_weights.T
        #print(f"--- End of Iteration {iteration_count} ---")
        #iteration_count += 1  # Increment iteration count
        
        
        
        
        # Save the view matrix to
        coint_end_day_index = df_ret.index.get_loc(coint_end_day)
        if coint_end_day_index+rolling_days >= len(df_ret.index):
            print("Error: coint_start_day index is out of range")
        # Handle the error in an appropriate way
            break
        
    
        
        #rebalance_day = df_ret2.index[coint_end_day_index + rolling_days]
        rebalance_day = df_ret.index[coint_end_day_index + rolling_days] 
        #rebalance_asset = df_ret2.loc[rebalance_day]
        rebalance_asset = df_ret.loc[coint_end_day:rebalance_day]
        rebalance_asset_return = (1+rebalance_asset).prod()-1
        portfolio_value3= np.dot(np.array(rebalance_asset_return.T), np.array(normalized_weights3.iloc[:, 1]))
        portfolio_values3.append(portfolio_value3.tolist())
        #print(portfolio_value3)
        
        start_day = df_ret.index[end_day_index - t4] if end_day_index - t4>= 0 else df_ret.index[0]
        
        if (end_day_index + rolling_days) < num:
            end_day = df_ret.index[end_day_index]
        else:
            end_day = df_ret.index[num-1]
        ret_subset = df_ret.loc[start_day:end_day]
        #ret_subset = df_ret_full.loc[coint_start_day:coint_end_day]
        
        # Calculate the covariance matrix for the current iteration
        cov_matrix = ret_subset.cov()
        csv_file_path1 = os.path.join(folder_path1, f"output_{iteration_count}.csv")
        Q4 = pd.read_csv(csv_file_path1)["ols Monte Carlo Simulation modified"]

        csv_file_path2 = os.path.join(folder_path2, f"pview_df_iteration_{iteration_count}.csv")
        P = pd.read_csv(csv_file_path2)
        #print(P)
        omega = tau * np.dot(np.dot(P,cov_matrix), P.T)
        #print(omega)
        bl_model4 = BlackLittermanModel(cov_matrix,Q=Q4,P=P,omega=omega)
        rets4 = bl_model4.bl_returns()
        ef4 = EfficientFrontier(rets4,cov_matrix,weight_bounds=(-1, 1))
        #print(ef.min_volatility())
        bl4 = ef4.max_sharpe(risk_free_rate = -1)  
        #bl_model4.bl_weights()  
        #bl4 = bl_model4.clean_weights()
        bl4_df = pd.DataFrame.from_dict(bl4, orient='index', columns=['Weight'])

        #print(bl4_df)
        total_weight4 = sum(bl4.values())
        normalized_weights4 = pd.DataFrame((key, value/total_weight4) for key, value in bl4.items())
        #print(normalized_weights)
    #df_ret2.iloc[44,:]*normalized_weights.T
        #print(f"--- End of Iteration {iteration_count} ---")
        #iteration_count += 1  # Increment iteration count
        
        
        
        
        # Save the view matrix to
        coint_end_day_index = df_ret.index.get_loc(coint_end_day)
        if coint_end_day_index+rolling_days >= len(df_ret.index):
            print("Error: coint_start_day index is out of range")
        # Handle the error in an appropriate way
            break
        
    
        
        #rebalance_day = df_ret2.index[coint_end_day_index + rolling_days]
        rebalance_day = df_ret.index[coint_end_day_index + rolling_days] 
        #rebalance_asset = df_ret2.loc[rebalance_day]
        rebalance_asset = df_ret.loc[coint_end_day:rebalance_day]
        rebalance_asset_return = (1+rebalance_asset).prod()-1
        portfolio_value4= np.dot(np.array(rebalance_asset_return.T), np.array(normalized_weights4.iloc[:, 1]))
        portfolio_values4.append(portfolio_value4.tolist())
        #print(portfolio_value4)
        
        start_day = df_ret.index[end_day_index - t5] if end_day_index - t5>= 0 else df_ret.index[0]
        
        if (end_day_index + rolling_days) < num:
            end_day = df_ret.index[end_day_index]
        else:
            end_day = df_ret.index[num-1]
        ret_subset = df_ret.loc[start_day:end_day]
        #ret_subset = df_ret_full.loc[coint_start_day:coint_end_day]
        
        # Calculate the covariance matrix for the current iteration
        cov_matrix = ret_subset.cov()
        csv_file_path1 = os.path.join(folder_path1, f"output_{iteration_count}.csv")
        Q5 = pd.read_csv(csv_file_path1)["kf Monte Carlo Simulation t"]

        csv_file_path2 = os.path.join(folder_path2, f"pview_df_iteration_{iteration_count}.csv")
        P = pd.read_csv(csv_file_path2)
        #print(P)
        omega = tau * np.dot(np.dot(P,cov_matrix), P.T)
        #print(omega)
        bl_model5 = BlackLittermanModel(cov_matrix,Q=Q5,P=P,omega=omega)
        rets5 = bl_model5.bl_returns()
        ef5 = EfficientFrontier(rets5,cov_matrix,weight_bounds=(-1, 1))
        #print(ef.min_volatility())
        bl5 = ef5.max_sharpe(risk_free_rate = -1)  
        #bl_model5.bl_weights()  
        #bl5 = bl_model5.clean_weights()
        bl5_df = pd.DataFrame.from_dict(bl5, orient='index', columns=['Weight'])

        #print(bl5_df)
        total_weight5 = sum(bl5.values())
        normalized_weights5 = pd.DataFrame((key, value/total_weight5) for key, value in bl5.items())
        #print(normalized_weights)
    #df_ret2.iloc[44,:]*normalized_weights.T
        #print(f"--- End of Iteration {iteration_count} ---")
        #iteration_count += 1  # Increment iteration count
        
        
        
        
        # Save the view matrix to
        coint_end_day_index = df_ret.index.get_loc(coint_end_day)
        if coint_end_day_index+rolling_days >= len(df_ret.index):
            print("Error: coint_start_day index is out of range")
        # Handle the error in an appropriate way
            break
        
    
        
        #rebalance_day = df_ret2.index[coint_end_day_index + rolling_days]
        rebalance_day = df_ret.index[coint_end_day_index + rolling_days] 
        #rebalance_asset = df_ret2.loc[rebalance_day]
        rebalance_asset = df_ret.loc[coint_end_day:rebalance_day]
        rebalance_asset_return = (1+rebalance_asset).prod()-1
        portfolio_value5= np.dot(np.array(rebalance_asset_return.T), np.array(normalized_weights5.iloc[:, 1]))
        portfolio_values5.append(portfolio_value5.tolist())
        #print(portfolio_value5)
        
        start_day = df_ret.index[end_day_index - t6] if end_day_index - t6>= 0 else df_ret.index[0]
        
        if (end_day_index + rolling_days) < num:
            end_day = df_ret.index[end_day_index]
        else:
            end_day = df_ret.index[num-1]
        ret_subset = df_ret.loc[start_day:end_day]
        #ret_subset = df_ret_full.loc[coint_start_day:coint_end_day]
        
        # Calculate the covariance matrix for the current iteration
        cov_matrix = ret_subset.cov()
        csv_file_path1 = os.path.join(folder_path1, f"output_{iteration_count}.csv")
        Q6 = pd.read_csv(csv_file_path1)["kf Monte Carlo Simulation modified"]

        csv_file_path2 = os.path.join(folder_path2, f"pview_df_iteration_{iteration_count}.csv")
        P = pd.read_csv(csv_file_path2)
        #print(P)
        omega = tau * np.dot(np.dot(P,cov_matrix), P.T)
        #print(omega)
        bl_model6 = BlackLittermanModel(cov_matrix,Q=Q6,P=P,omega=omega)
        rets6 = bl_model6.bl_returns()
        ef6 = EfficientFrontier(rets6,cov_matrix,weight_bounds=(-1, 1))
        #print(ef.min_volatility())
        bl6 = ef6.max_sharpe(risk_free_rate = -1)  
        #bl_model6.bl_weights()  
        #bl6 = bl_model6.clean_weights()
        bl6_df = pd.DataFrame.from_dict(bl6, orient='index', columns=['Weight'])

        #print(bl6_df)
        total_weight6 = sum(bl6.values())
        normalized_weights6 = pd.DataFrame((key, value/total_weight6) for key, value in bl6.items())
        #rebalance_day = df_ret2.index[coint_end_day_index + rolling_days]
        rebalance_day = df_ret.index[coint_end_day_index + rolling_days] 
        #rebalance_asset = df_ret2.loc[rebalance_day]
        rebalance_asset = df_ret.loc[coint_end_day:rebalance_day]
        rebalance_asset_return = (1+rebalance_asset).prod()-1
        portfolio_value6= np.dot(np.array(rebalance_asset_return.T), np.array(normalized_weights6.iloc[:, 1]))
        portfolio_values6.append(portfolio_value6.tolist())
        #print(portfolio_value6)
        #print(normalized_weights)
    #df_ret2.iloc[44,:]*normalized_weights.T
        #print(f"--- End of Iteration {iteration_count} ---")
        #iteration_count += 1  # Increment iteration count
        
        
        
        
        # Save the view matrix to
        coint_end_day_index = df_ret.index.get_loc(coint_end_day)
        if coint_end_day_index+rolling_days >= len(df_ret.index):
            print("Error: coint_start_day index is out of range")
        # Handle the error in an appropriate way
            break
        
        
        
    
        
        #rebalance_day = df_ret2.index[coint_end_day_index + rolling_days]
        #rebalance_day = df_ret.index[coint_end_day_index + rolling_days] 
        #rebalance_asset = df_ret2.loc[rebalance_day]
        #rebalance_asset = df_ret.loc[coint_end_day:rebalance_day]
        #rebalance_asset_return = (1+rebalance_asset).prod()-1
        #portfolio_value = np.dot(np.array(rebalance_asset_return.T), np.array(normalized_weights1.iloc[:, 1]))
        #portfolio_values.append(portfolio_value.tolist())
        #print(portfolio_value)
        new_data = {'Iteration': iteration_count,
                    'diff mc t': portfolio_value1,
                    'diff Monte Carlo Simulation modified': portfolio_value2,
                    'ols mc t': portfolio_value3,
                    'ols Monte Carlo Simulation modified': portfolio_value4,
                    'kf mc t': portfolio_value5,
                    'kf Monte Carlo Simulation modified': portfolio_value6}
                        #'Actual Residuals': actual_residual}
            
        new_row_df = pd.DataFrame([new_data])

        # Concatenate the new DataFrame with the existing 'results_df'
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)

            #results_df = results_df.append(new_data, ignore_index=True)
        #print(results_df)

        print(f"--- End of Iteration {iteration_count} ---")
        iteration_count += 1  # Increment iteration count
        
        coint_start_day = df_ret.index[df_ret.index.get_loc(coint_start_day) + rolling_days]
        coint_end_day_index = df_ret.index.get_loc(coint_end_day) +rolling_days
        if coint_end_day_index < num:
            coint_end_day = df_ret.index[coint_end_day_index]
            end_day_index = coint_end_day_index 
        else:
            end_day_index = coint_end_day_index


    return results_df
