import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt


days = 100
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=days)

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

def get_data(stock, start, end):
    """
    (list, datetime, datetime) -> pandas dataframe
    This function takes the list of stocks as well as a length of time
    and returns the stock data for the tickers in the list

    input: ['MFC.TO', 'BTO.TO']
    """
    # Get Data from yahoo finance
    stockdata = pdr.get_data_yahoo(stock, start=start, end=end)

    # Choose only the stock variable
    stockdata = stockdata['Close']
    return stockdata

def daily_return_array(stock_data):
    returns = np.array(np.add(stock_data.pct_change(), 1))
    return returns

def get_paths(returns):
    mega_x_list = []
    mega_y_list = []
    nbr_paths = 1000
    nbr_days = 100
    length_array = len(returns)

    for i in range(nbr_paths):
        x_list = [0]
        y_list = [1]
        y = 1
        for x in range(nbr_days):
            index = np.random.randint(0, length_array-1)
            y = y*returns[index]
            y_list.append(y)
            x_list.append(x)
        mega_x_list.append(x_list)
        mega_y_list.append(y_list)
    return mega_x_list, mega_y_list

def show_paths(mega_x_list, mega_y_list):
    g = plt.figure(1)
    for i in range(len(mega_x_list)):
        plt.plot(mega_x_list[i], mega_y_list[i])
    #g.show()

def get_end_value(mega_x_list, mega_y_list):
    list_end = np.array([item[-1] for item in mega_y_list])
    return list_end

def show_hist(mega_y_list, nbr_bins):
    list_end = get_end_value(mega_x_list, mega_y_list)
    f = plt.figure(2)
    plt.hist(list_end, bins=nbr_bins)
    #f.show()

stock_data = get_data(['ATD.TO'], start_date, end_date)
returns = daily_return_array(stock_data)
mega_x_list, mega_y_list = get_paths(returns)
show_paths(mega_x_list, mega_y_list)
nbr_bins = 25
show_hist(mega_y_list, nbr_bins)
plt.show()