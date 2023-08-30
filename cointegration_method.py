import statsmodels.api as sm
from itertools import combinations

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






