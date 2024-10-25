# from flask import Flask, render_template
# from model import load_data, train_model, make_prediction, plot_prediction, calculate_metrics

# app = Flask(__name__)

# @app.route('/')
# def index():
#     stock_data = load_data()
#     fitted_model = train_model(stock_data)
#     result, conf_ins = make_prediction(fitted_model)
#     plot_url = plot_prediction(stock_data, result, conf_ins)
#     metrics = calculate_metrics(stock_data['Close'].values[-51:], result)
    
#     return render_template('index.html', plot_url=plot_url, metrics=metrics)

# if __name__ == '__main__':
#     app.run(debug=True)



# app.py

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and prepare the data
stock_data = pd.read_csv('Cleaned_ACGL_data2.csv', parse_dates=['Date'], index_col='Date')

# Train the model
model = SARIMAX(stock_data['Close'].values, order=(0, 1, 0), trend='c')
fitted_model = model.fit(disp=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None
    plot_url = None
    
    if request.method == 'POST':
        days = int(request.form['days'])
        forecast = fitted_model.forecast(days)
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data.index, stock_data['Close'], label='Historical Data')
        plt.plot(pd.date_range(start=stock_data.index[-1], periods=days+1, freq='D')[1:], 
                 forecast, label='Forecast')
        plt.title('ARCH CAPITAL GROUP Stock Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        
        # Save plot to a binary stream
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        
        plt.close()  # Close the plot to free up memory
    
    return render_template('index.html', forecast=forecast, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)