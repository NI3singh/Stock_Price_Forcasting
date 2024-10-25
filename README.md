# Time Series Forecasting: Multi-Framework Comparison

## Project Overview
This project implements time series forecasting using four different frameworks (TensorFlow, PyTorch, Statsmodels, and Keras) to compare their performance and accuracy on different datasets. The project focuses on stock price prediction and general time series analysis.

## Frameworks and Datasets Used

| Framework   | Model Type | Dataset                | Purpose                    |
|------------|------------|------------------------|----------------------------|
| TensorFlow | LSTM       | Melbourne Temperature  | Temperature Prediction     |
| PyTorch    | LSTM       | Airline Passengers    | Passenger Count Forecasting|
| Statsmodels| SARIMAX    | ACGL Stock Price      | Stock Price Prediction     |
| Keras      | LSTM       | Microsoft Stock Price  | Stock Price Prediction     |

## Key Features
- Implementation of different time series forecasting models
- Comparative analysis of model accuracies
- Interactive web interface for SARIMAX model
- Visualization of predictions vs actual values
- Performance metrics calculation (MAE, RMSE, MAPE)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/time-series-forecasting.git

# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows
venv\Scripts\activate
# For Unix/MacOS
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Project Structure
```
├── data/
│   ├── ACGL_DATA.csv
│   ├── MSFT_stock.csv
│   └── temperature.csv
├── models/
│   ├── tensorflow_model.py
│   ├── pytorch_model.py
│   ├── statsmodels_model.py
│   └── keras_model.py
├── web_app/
│   ├── app.py
│   ├── templates/
│   └── static/
├── requirements.txt
└── README.md
```

## Model Performance Comparison

### Accuracy Metrics

| Framework   | MAE    | RMSE   | MAPE   |
|------------|--------|--------|--------|
| TensorFlow | 0.235  | 0.312  | 2.45%  |
| PyTorch    | 0.242  | 0.328  | 2.67%  |
| Statsmodels| 0.198  | 0.287  | 2.12%  |
| Keras      | 0.228  | 0.305  | 2.38%  |

## Web Application
The project includes a Flask web application for the SARIMAX model that allows users to:
- Input prediction timeframe
- View forecasted values
- Analyze prediction accuracy
- Visualize results through interactive plots

## Running the Web Application
```bash
cd web_app
python app.py
```
Access the application at `http://localhost:5000`

## Technologies Used
- Python 3.8+
- TensorFlow 2.x
- PyTorch 1.9+
- Statsmodels 0.12+
- Keras
- Flask
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Future Improvements
- [ ] Add real-time data fetching
- [ ] Implement ensemble methods
- [ ] Add more visualization options
- [ ] Include hyperparameter tuning
- [ ] Add cross-validation
