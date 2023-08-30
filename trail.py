import numpy as np
import scipy.stats as stats
import pandas as pd

def calculate_portfolio_statistics(portfolio_df, risk_free_rate=0.02, target_return=0.05, confidence_level=0.95):
    first_column_label = portfolio_df.columns[0]
    returns_series = portfolio_df[first_column_label]
    #returns_series = portfolio_df['0']
    
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
    
    max_drawdown = (returns_series / (returns_series.cummax()) - 1).min()
    turnover = np.mean(np.abs(returns_series.diff().dropna()))
    
    # Assuming risk-free rate is 2% annually
    sharpe_ratio = (annualized_mean_return - risk_free_rate) / volatility
    
    downside_returns = returns_series[returns_series < target_return]
    downside_volatility = downside_returns.std() * np.sqrt(12)
    sortino_ratio = (annualized_mean_return - risk_free_rate) / downside_volatility
    
    # Calculate Value at Risk (VaR) and Expected Shortfall (ES)
    VaR = -np.percentile(returns_series, 100 * (1 - confidence_level))
    ES = -downside_returns.mean()
    calmar_ratio = annualized_mean_return / abs(max_drawdown)
    # Calculate Mean t-statistic
    mean_t_statistic = stats.ttest_1samp(returns_series, popmean=monthly_mean_return)[0]
    
    # Calculate the percentage of observations less than 0
    observations_less_than_zero = (returns_series < 0).mean()
    
    # Return the results as a dictionary
    portfolio_statistics = {
        "Annualized Mean Return": annualized_mean_return,
        "Volatility (Annualized Std. Dev.)": volatility,
        "Mean t-statistic": mean_t_statistic,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Min": min_return,
        "Max": max_return,
        "% Observations < 0": observations_less_than_zero,
        "Maximum Drawdown": max_drawdown,
        "Turnover": turnover,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        f"Value at Risk (VaR) at {confidence_level*100:.2f}% confidence level": VaR,
        f"Expected Shortfall (ES) at {confidence_level*100:.2f}% confidence level": ES
    }
    
    return portfolio_statistics

portfoliovalues = [-4.007253195289776
-0.03849407280689909,
-0.061396828730537846,
0.07848125612381546,
0.2997511791227188,
0.36173421993052646,
0.24053071794272335,
0.5547781163048007,
0.1611260034621615,
0.09743514132624198,
1.6281243926002393]

portfolio_df = pd.DataFrame(portfoliovalues)

x = calculate_portfolio_statistics(portfolio_df, risk_free_rate=0.02, target_return=0.05, confidence_level=0.95)
print(x)