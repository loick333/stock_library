import efficient_frontier as ef
import datetime as dt

days = ef.get_time_delta()
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=days)

nbr_test = ef.get_nbr_tests()

stock_list = ef.get_list_stocks()

stockdata = ef.get_data(stock_list, start_date, end_date)
gmean_returns = ef.calculate_geo_mean(stockdata)[0]
cov_matrix = ef.calculate_geo_mean(stockdata)[1]

list_rand_weights = ef.get_random_weights(nbr_test, stock_list)

list_tuples = ef.make_list_tuples(end_date, start_date, nbr_test, stockdata, gmean_returns, cov_matrix, list_rand_weights)
ef.make_scatterplot_returns_var(list_tuples)