import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate random stock data for 10 years (2520 days, assuming 252 trading days per year)
np.random.seed(42)
num_assets = 5
num_days = 2520
returns = np.random.randn(num_days, num_assets)

# Create a DataFrame of random returns
tickers = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN']
returns_df = pd.DataFrame(returns, columns=tickers)

# Calculate the annualized mean returns and covariance matrix
mean_returns = returns_df.mean() * 252
cov_matrix = returns_df.cov() * 252

# Define the portfolio performance (returns, volatility, Sharpe ratio)
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    returns = np.sum(weights * mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe_ratio

# Objective function: Minimize portfolio variance (risk)
def minimize_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Constraints: Sum of weights = 1 (fully invested)
def weight_constraint(weights):
    return np.sum(weights) - 1

# Bounds for each asset (0 <= weight <= 1)
bounds = tuple((0, 1) for asset in range(num_assets))

# Initial guess (equal distribution)
initial_weights = num_assets * [1. / num_assets]

# Constraints for optimization
constraints = ({'type': 'eq', 'fun': weight_constraint})

# Perform the optimization to minimize variance
optimized_result = minimize(minimize_variance, initial_weights, args=cov_matrix, 
                            method='SLSQP', bounds=bounds, constraints=constraints)

# Optimized portfolio weights
optimized_weights = optimized_result.x

# Calculate optimized portfolio performance
opt_returns, opt_volatility, opt_sharpe = portfolio_performance(optimized_weights, mean_returns, cov_matrix)

# Visualization: Plot the Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.01):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_volatility, _ = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe Ratio

    return results, weights_record

# Generate the Efficient Frontier
results, weights_record = efficient_frontier(mean_returns, cov_matrix)

# Plot the Efficient Frontier
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='YlGnBu', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')

# Plot the optimized portfolio
plt.scatter(opt_volatility, opt_returns, marker='*', color='r', s=200, label='Optimized Portfolio')
plt.legend(labelspacing=0.8)
plt.show()

# Output the optimized weights and performance metrics
print("Optimized Weights:", optimized_weights)
print("Expected Annual Return:", opt_returns)
print("Annual Volatility (Risk):", opt_volatility)
print("Sharpe Ratio:", opt_sharpe)
