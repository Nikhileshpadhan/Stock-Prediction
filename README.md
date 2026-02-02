🤖 MarketMind AI: Research-Level Stock Predictor

MarketMind AI is a hybrid financial intelligence system designed to forecast stock price movements by fusing Time-Series Deep Learning with Human Sentiment Analysis.
Unlike traditional models that only look at historical charts, MarketMind simulates the impact of market hype or panic to provide a more holistic prediction.

📈 Performance Highlights

Based on the latest evaluation of the neural network, the model demonstrates high predictive accuracy:

R-squared (R²): 0.8189 — Explains ~82% of the variance in stock prices.

Root Mean Squared Error (RMSE): 7.7494 — Reflects the standard deviation of residuals.

Mean Absolute Error (MAE): 6.3525 — Average magnitude of prediction errors.

🌟 Key Features

Hybrid Intelligence: Combines LSTM layers for technical trends with a Dense Sentiment Layer for qualitative inputs.

Real-Time Data Pipeline: Streams live OHLCV market data via the Polygon.io REST API.

Dynamic Sentiment Adjuster: Users can inject sentiment scores (-1.0 to 1.0) to simulate market reactions.

Interactive Visuals: High-fidelity candlestick charts with 10-day and 50-day moving averages.

🧠 System Architecture

The core engine (marketmind_v1.keras) uses a Multi-Input Functional API:

Price Branch: 60-day window of price & volume data processed through 64-unit LSTM layers.

Sentiment Branch: 16-unit Dense layer processes sentiment scores as a bias/accelerant.

Concatenation Layer: Merges both branches to output a predicted Close price for the next trading session.

🛠️ Installation & Setup
Prerequisites

Python 3.9+

Polygon.io API Key

Setup Environment
pip install streamlit pandas numpy tensorflow joblib polygon-api-client plotly

📁 File Structure
├── app.py                  # Main application logic
├── assets/
│   ├── marketmind_v1.keras # The Hybrid Neural Network
│   └── price_scaler.pkl    # Normalization object (MinMaxScaler)

🚀 Usage

Configure API: Set your Polygon API key in app.py or as an environment variable.

Launch App:

streamlit run app.py

📌 Future Enhancements

Expand sentiment analysis to include Twitter/Reddit feeds.

Add ensemble models for multi-stock predictions.

Deploy as a cloud-based dashboard for real-time trading insights.
