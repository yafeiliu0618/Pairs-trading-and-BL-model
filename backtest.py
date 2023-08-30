import numpy as np
import scipy.stats as stats

def calculate_portfolio_statistics(portfolio_df, risk_free_rate=0.02, target_return=0, confidence_level=0.95):
    #first_column_label = portfolio_df.columns[0]
    #returns_series = portfolio_df[first_column_label]
    #returns_series = portfolio_df['0']
    returns_series = portfolio_df
    
    # Calculate portfolio statistics
    monthly_mean_return = returns_series.mean()
    annualized_mean_return = (1 + monthly_mean_return) ** 12 - 1
    volatility = returns_series.std() * np.sqrt(12)
     # Calculate standard deviation, skewness, and kurtosis
    skewness = returns_series.skew()
    kurtosis = returns_series.kurtosis()

    
    # Calculate Min and Max
    min_return = returns_series.min()
    max_return = returns_series.max()
    
    # Calculate the percentage of observations less than 0
    observations_less_than_zero = (returns_series < 0).mean()
    
    #max_drawdown = (returns_series / (returns_series.cummax()) - 1).min()
    cumulative_returns = (1 + returns_series).cumprod()

# Calculate cumulative maximum returns
    cumulative_max_returns = cumulative_returns.cummax()
    
# Calculate drawdown series
    drawdown = (cumulative_returns - cumulative_max_returns)/cumulative_max_returns
    #drawdown = returns_series.div(returns_series.cummax()).sub(1)
# Calculate maximum drawdown
    max_drawdown = abs(drawdown.min())
    #turnover = np.mean(np.abs(returns_series.diff().dropna()))
    annualized_return = (1+returns_series)**12-1
    # Assuming risk-free rate is 2% annually
    sharpe_ratio = (annualized_mean_return - risk_free_rate) / volatility
    
    downside_returns = returns_series[returns_series < target_return]
    downside_volatility = downside_returns.std() * np.sqrt(12)
    sortino_ratio = (annualized_mean_return - risk_free_rate) / downside_volatility
    excess_returns = np.array(annualized_return) - risk_free_rate
    negative_returns = np.minimum(returns_series, 0)
    VaR = -np.percentile(negative_returns, (1-confidence_level) * 100)
    ES = -np.mean(negative_returns[negative_returns <= -VaR])
    # Calculate Value at Risk (VaR) and Expected Shortfall (ES)
    #VaR = -np.percentile(returns_series, 100 * (1 - confidence_level))
    #ES = -downside_returns.mean()
    starr_ratio = (annualized_mean_return - risk_free_rate)/ ES
    #starr_ratio = np.mean(excess_returns)/ ES
    #calmar_ratio =  np.mean(excess_returns)/ abs(max_drawdown)
    calmar_ratio =  annualized_mean_return/ abs(max_drawdown)
    # Calculate Mean t-statistic
    #mean_t_statistic = stats.ttest_1samp(returns_series, popmean=monthly_mean_return)[0]
    
    # Calculate the percentage of observations less than 0
    observations_less_than_zero = (returns_series < 0).mean()
    
    # Return the results as a dictionary
    portfolio_statistics = {
        "Annualized Mean Return": annualized_mean_return,
        "Volatility (Annualized Std. Dev.)": volatility,
        #"Mean t-statistic": mean_t_statistic,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Min": min_return,
        "Max": max_return,
        "% Observations < 0": observations_less_than_zero,
        "Maximum Drawdown": max_drawdown,
        "Star Ratio":starr_ratio,
        #"Turnover": turnover,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        f"Value at Risk (VaR) at {confidence_level*100:.2f}% confidence level": VaR,
        f"Expected Shortfall (ES) at {confidence_level*100:.2f}% confidence level": ES
    }
    
    return portfolio_statistics




