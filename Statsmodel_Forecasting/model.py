# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.tsa.seasonal import seasonal_decompose
# from pmdarima.arima import auto_arima
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import math
# import io
# import base64

# def load_data():
#     stock_data = pd.read_csv(
#         'Cleaned_ACGL_data2.csv', 
#         sep=',', 
#         index_col='Date', 
#         parse_dates=['Date'], 
#         date_parser=lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
#     ).fillna(0)
#     return stock_data

# def train_model(stock_data):
#     model = sm.tsa.statespace.SARIMAX(stock_data['Close'].values[:-51], trend='c', order=(0,1,0))
#     fitted = model.fit(disp=False)
#     return fitted

# def make_prediction(fitted, steps=51):
#     result = fitted.forecast(steps, alpha=0.05)
#     conf_ins = fitted.get_forecast(steps).summary_frame()
#     return result, conf_ins

# def plot_prediction(stock_data, result, conf_ins):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(stock_data.index, stock_data['Close'].values, label='Actual Stock Price')
#     ax.plot(stock_data.index[-51:], result, label='Predicted Stock Price')
#     ax.plot(stock_data.index[-51:], conf_ins['mean_ci_lower'], label='Lower CI')
#     ax.plot(stock_data.index[-51:], conf_ins['mean_ci_upper'], label='Upper CI')
#     ax.legend()
#     fig.autofmt_xdate()
#     plt.title('ARCH CAPITAL GROUP Stock Price Prediction')
#     plt.xlabel('Time')
#     plt.ylabel('ARCH CAPITAL GROUP Stock Price')
    
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plot_url = base64.b64encode(img.getvalue()).decode()
#     plt.close()
#     return plot_url

# def calculate_metrics(actual, predicted):
#     mse = mean_squared_error(actual, predicted)
#     mae = mean_absolute_error(actual, predicted)
#     rmse = math.sqrt(mse)
#     mape = np.mean(np.abs(predicted - actual)/np.abs(actual))
#     return {
#         'MSE': mse,
#         'MAE': mae,
#         'RMSE': rmse,
#         'MAPE': mape
#     }