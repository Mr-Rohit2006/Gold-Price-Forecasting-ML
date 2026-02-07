project:
  name: "Gold Price Forecasting using Machine Learning"
  repository: "https://github.com/Mr-Rohit2006/Gold-Price-Forecasting-ML"
  type: "Machine Learning | Time-Series Forecasting"
  description: >
    This project is a machine learning–based gold price forecasting system.
    It automatically fetches the latest gold market data and predicts the
    next-day gold price using time-series features and regression techniques.
    The model updates itself every time it is run.

features:
  - Automatic fetching of live gold price data
  - Time-series feature engineering using lag values
  - Next-day gold price prediction
  - Graphical visualization of past prices and predicted value
  - No manual data update required

tech_stack:
  language: "Python"
  libraries:
    - pandas
    - numpy
    - scikit-learn
    - yfinance
    - matplotlib
  model: "Linear Regression"
  domain: "Time-Series Forecasting"

project_structure:
  - gold_next_day_prediction.py
  - Indian_Gold_Price_per_Gram_Till_Today.csv
  - README.yml

working:
  steps:
    - Fetch latest gold price data from online source
    - Convert price to INR per gram
    - Generate lag-based time-series features
    - Train regression model on recent data
    - Predict next-day gold price
    - Display prediction with trend graph

how_to_run:
  clone_repository:
    command: "git clone https://github.com/Mr-Rohit2006/Gold-Price-Forecasting-ML.git"
  navigate_to_project:
    command: "cd Gold-Price-Forecasting-ML"
  install_dependencies:
    command: "pip install pandas numpy scikit-learn yfinance matplotlib"
  run_project:
    command: "python gold_next_day_prediction.py"

sample_output:
  today_gold_price_10g: "₹132123.27"
  predicted_next_day_price_10g: "₹132300.86"
  note: "A graph is displayed showing recent prices and the predicted value."

model_performance:
  evaluation_type: "Regression Metrics"
  mean_absolute_error_per_gram: "₹20–₹40"
  approximate_error_for_10g: "₹200–₹400"
  directional_accuracy: "~65%"
  remark: >
    The model is suitable for educational and trend analysis purposes,
    not for real-world trading.

learning_outcomes:
  - Understanding time-series forecasting concepts
  - Working with live financial data
  - Feature engineering using lag values
  - Applying regression models to real-world problems
  - Data visualization and interpretation

future_enhancements:
  - ARIMA or LSTM based forecasting
  - Multi-day price prediction
  - Inclusion of macroeconomic indicators
  - Web dashboard for real-time predictions
  - Automated daily prediction alerts

author:
  name: "Rohit Kumar"
  github: "https://github.com/Mr-Rohit2006"

note: >
  If you find this project useful, consider giving it a star on GitHub.
