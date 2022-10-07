import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from spicy import stats
from pandas import DataFrame
import matplotlib.pyplot as plt
import random

#random.seed(6)
days = 600
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=days)
stocks = ['MFC.TO', 'DOL.TO', 'BTO.TO', 'BLU.TO']
#percentage = DataFrame([1, 0])
nbr_tests = 10000

def get_time_delta():
    """
    () -> list
    This function asks the user for the period of time for which
    he would like to get the data.
    """
    input_days = int(input("Please enter the number of days you would like to retrieve stock data for:"))
    return input_days

def get_nbr_tests():
    """
    () -> int
    This function asks the user for the number of tests he would like to perform
    """
    nbr_test = round(input("Please enter the number of tests you would like to perform: "))
    return nbr_test

def get_list_stocks():
    """
    () -> list
    This function asks the user for a list of stocks
    and returns a list of those stocks.

    input: MFC, BTO
    output: ['MFC.TO', 'BTO.TO']
    """
    print("Please list the Canadian stock tickers you would like to proceed with."
          "Example: BTO, MFC, ATD")
    input_list = input("List of stocks:").replace(" ", "").split(',')
    stock_list = [stock + '.TO' for stock in input_list]
    print(stock_list)
    return stock_list

def get_data(stocks, start, end):
    """
    (list, datetime, datetime) -> pandas dataframe
    This function takes the list of stocks as well as a length of time
    and returns the stock data for the tickers in the list

    input: ['MFC.TO', 'BTO.TO']
    """
    # Get Data from yahoo finance
    stockdata = pdr.get_data_yahoo(stocks, start=start, end=end)

    # Choose only the stock variable
    stockdata = stockdata['Close']
    return stockdata

def calculate_geo_mean(stockdata):
    """
    (pandas dataframe) -> pandas dataframe

    This function takes a pandas dataframe of a number of stock tickers with their respective close dates
    to output the geometric mean of %change per day. The formula for the geometric mean is
    Product of [1 + daily return]^(1/n) - 1.
    """
    # Calculate Percent Change
    returns = np.add(stockdata.pct_change(),1)
    cov_matrix = returns.cov()

    # Select all rows except first one since first index is Not available because it is percent change
    # Also add one to compute geometric mean later
    returns = returns.iloc[1:, :]

    # Get geometric mean for percent change and substract 1
    gmean_returns_list = np.subtract(stats.gmean(returns.loc[:,]), 1)
    gmean_returns_df = {'g_mean': gmean_returns_list}
    gmean_returns_df = DataFrame(gmean_returns_df, index=stocks) #add , index=stocks for indexes

    # Turn pandas dataframe into a pandas series so as to multiply it later like a matrix
    gmean_returns_df_transpose = gmean_returns_df.T
    gmean_returns_series = gmean_returns_df_transpose.squeeze(axis=0)
    return gmean_returns_series, cov_matrix

def get_portfolio_return(percentage, gmean_returns):
    gmean_returns = gmean_returns.add(1)
    return_array = np.dot(gmean_returns.T, percentage)
    portfolio_return = return_array**365
    return portfolio_return

def get_portfolio_variance(percentage, cov_matrix):
    """
    (list, pandas.data.frame) -> float

    This function takes a list of percentage which weights the portfolio and the covariance matrice
    """
    # Change percentage List into a dataframe for further matrix manipulation
    percentage_df = DataFrame(percentage)

    # Formula for portfolio variance where P is the percentage matrix (2x1) and
    # C the covariance Matrix (2x2) and ^T the transpose: [P]^T[C][P]
    portfolio_variance = float(np.sqrt(np.dot(percentage_df.T, np.dot(cov_matrix, percentage_df))))
    return portfolio_variance

def get_random_weights(nbr_tests, stocks):
    """
    (int, list) -> list

    This function takes a integer and a list as input. The integer represents the number of different weights
    we would like to test our portfolio with. The list represents a different stocks we would like to combine.
    The function outputs a list of lists.
    """

    nbr_stocks = len(stocks)
    list_of_list_weights = []
    w_max = 1

    # First loop until we reach the number of different weights we will test our portfolio with
    for i in range(nbr_tests):
        list_weights = []

        # Second loop, we generate a random number and add it to the list
        for j in range(nbr_stocks - 1):

            # w_max represents the maximum value the weight can take since the sum of weights cannot be greater than 1
            # Every time a weight is added w_mx is reduced by that amount
            weight_i = round(random.uniform(0, w_max), 2)
            w_max -= weight_i

            # Add wight to list
            list_weights.append(weight_i)

        # The last weigh is simply 1 minus the sunm of the other weights, which adds to 1 over the whole list
        list_weights.append(round(1-sum(list_weights),2))
        list_of_list_weights.append(list_weights)

        # Reset w_max to 1 for the new list
        w_max = 1
    return list_of_list_weights

def make_list_tuples(end_date, start_date, nbr_test, stockdata, gmean_returns, cov_matrix, list_rand_weights):
    # Get start date and end date
    end = end_date
    start = start_date
    nbr_tests = nbr_test

    # Get all necessary variables
    stockdata = get_data(stocks, start, end)
    gmean_returns = calculate_geo_mean(stockdata)[0]
    cov_matrix = calculate_geo_mean(stockdata)[1]

    list_rand_weights = get_random_weights(nbr_tests, stocks)
    #
    list_returns_variance = []
    for weights in list_rand_weights:
        variance = get_portfolio_variance(weights, cov_matrix)
        returns = get_portfolio_return(weights, gmean_returns)
        tuple_var_returns = (returns, variance)
        list_returns_variance.append(tuple_var_returns)
    return list_returns_variance

def make_scatterplot_returns_var(list_tuples):
    x, y = zip(*list_tuples)
    plt.scatter(y, x)
    plt.show()
    return

end = end_date
start = start_date
stockdata = get_data(stocks, start, end)
gmean_returns = calculate_geo_mean(stockdata)[0]
cov_matrix = calculate_geo_mean(stockdata)[1]
list_rand_weights = get_random_weights(nbr_tests, stocks)

print(gmean_returns)

list_tuples = make_list_tuples(end_date, start_date, nbr_tests, stockdata, gmean_returns, cov_matrix, list_rand_weights)
make_scatterplot_returns_var(list_tuples)
