# Stock Prediction App

## Overview

The Stock Prediction App is a web application built using Plotly Dash and powered by machine learning models to predict stock prices based on historical data.

## Features

- **Interactive Dashboard**: Visualize historical stock data and predicted prices.
- **Machine Learning Models**: Includes LSTM-based models for stock price prediction.
- **Customizable Settings**: Adjust model parameters and input data for predictions.
- **Export and Save**: Download predictions or save them for future reference.

## Installation

### Requirements

Python 3.12

To run the Stock Prediction App locally, follow these steps:

### Using `venv` (Virtual Environment)

#### For Windows:

1. Clone the repository:

   ```bash
   git clone https://github.com/Dat-TG/Stock-Price-Prediction-DV.git
   cd Stock-Price-Prediction-DV
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   ```bash
   .\venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   python app.py
   ```

6. Open a web browser and go to `http://localhost:8050` to view the application.

#### For Linux:

1. Clone the repository:

   ```bash
   git clone https://github.com/Dat-TG/Stock-Price-Prediction-DV.git
   cd Stock-Price-Prediction-DV
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   python app.py
   ```

6. Open a web browser and go to `http://localhost:8050` to view the application.

## Usage

- **Input Data**: Upload historical stock data in CSV format.
- **Configure Model**: Adjust parameters like epochs, batch size, etc.
- **Generate Predictions**: Generate stock price predictions.
- **Visualize Results**: View predicted prices and compare with actual data on the dashboard.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests to suggest improvements or fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Built using [Plotly Dash](https://dash.plotly.com/)
- Machine learning models implemented with [Keras](https://keras.io/)
- Stock data provided by [Investing](https://investing.com/)

## Support

For any issues or questions related to the Stock Prediction App, please [open an issue](https://github.com/Dat-TG/Stock-Price-Prediction-DV/issues/new).

---
