import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from polygon import RESTClient
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING (With Caching) ---
@st.cache_resource
def load_assets():
    try:
        # Assumes your files are in the 'assets' folder
        model = tf.keras.models.load_model('assets/marketmind_v1.keras')
        scaler = joblib.load('assets/price_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

# --- 3. DATA FETCHING FUNCTION ---
def fetch_market_data(api_key, ticker):
    client = RESTClient(api_key)
    # Fetch 120 days to ensure we have a full 60-day window after holidays
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
    
    data = []
    # Using list_aggs for historical bars
    for agg in client.list_aggs(ticker, 1, "day", start_date, end_date):
        data.append(agg)
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- 4. MAIN APP INTERFACE ---

# Optional: hardcode your API key here (NOT recommended). Keep None to avoid embedding keys in code.
_POLYGON_API_KEY_HARDCODED = 'VChm7zJhrHCEsv1SKGNJ0FeWf6polP8C'  # e.g. 'pk_...'


def get_api_key_from_sources():
    """Retrieve API key from hardcoded var, Streamlit secrets, or environment variables. Never show it in the UI."""
    if _POLYGON_API_KEY_HARDCODED:
        return _POLYGON_API_KEY_HARDCODED
    if st.secrets.get('POLYGON_API_KEY'):
        return st.secrets.get('POLYGON_API_KEY')
    if os.environ.get('POLYGON_API_KEY'):
        return os.environ.get('POLYGON_API_KEY')
    return None


def main():
    st.title("🤖 MarketMind: AI Stock Predictor")
    st.markdown("---")

    # Sidebar controls (API key is loaded securely; UI message removed)

    ticker = st.sidebar.text_input("Stock Ticker", value='AAPL').upper()
    days_back = st.sidebar.slider('History (days)', min_value=60, max_value=365, value=120, step=30)
    manual_sentiment = st.sidebar.slider("News Sentiment (-1 to 1)", -1.0, 1.0, 0.0)
    ma_short = st.sidebar.selectbox('Short MA', [5, 10, 20], index=1)
    ma_long = st.sidebar.selectbox('Long MA', [50, 100, 200], index=0)
    predict_btn = st.sidebar.button('Generate AI Prediction')

    # Display model status
    model, scaler = load_assets()

    st.markdown('### Quick status')
    cols = st.columns(3)
    cols[0].write(f'**Ticker:** {ticker}')
    cols[1].write(f'**Model:** {'loaded' if model is not None else 'missing'}')
    cols[2].write(f'**Scaler:** {'loaded' if scaler is not None else 'missing'}')

    # API key presence (hidden)
    key_loaded = bool(get_api_key_from_sources())
    if key_loaded:
        st.success('API key available (hidden)')
    else:
        st.warning('API key not found. Add POLYGON_API_KEY to Streamlit Secrets or environment.')

    if predict_btn:
        api_key = get_api_key_from_sources()
        if not api_key:
            st.error('API key required. Add POLYGON_API_KEY to Streamlit Secrets or set it as an environment variable (or set _POLYGON_API_KEY_HARDCODED at top).')
            return

        if model is None:
            st.error('Model not available. Please add the model to assets/ and reload.')
            return

        with st.spinner(f'Fetching market data for {ticker}...'):
            try:
                df = fetch_market_data(api_key, ticker)
            except Exception as e:
                st.error(f'Error fetching data: {e}')
                return

        if df.empty or len(df) < 60:
            st.error('Insufficient market data (min 60 days). Check ticker and API permissions.')
            return

        # Prepare model input
        last_60 = df[['open', 'high', 'low', 'close', 'volume']].tail(60).values
        if scaler is None:
            st.warning('Scaler missing; predictions may not be accurate.')
            # Still try to run model if possible
            try:
                scaled = last_60  # best-effort
            except Exception as e:
                st.error(f'Preprocessing failed: {e}')
                return
        else:
            scaled = scaler.transform(last_60)

        price_input = np.expand_dims(scaled, axis=0)
        sentiment_input = np.array([[manual_sentiment]])

        try:
            pred_scaled = model.predict([price_input, sentiment_input])
            # inverse transform predicted close when we have scaler
            pred_price = None
            if scaler is not None:
                dummy = np.zeros((1, 5))
                dummy[0, 3] = pred_scaled[0][0]
                pred_price = scaler.inverse_transform(dummy)[0, 3]
            else:
                pred_price = float(pred_scaled[0][0])
        except Exception as e:
            st.error(f'Model inference failed: {e}')
            return

        current_price = df['close'].iloc[-1]
        change = pred_price - current_price
        pct_change = (change / current_price) * 100

        # Display key metrics (confidence removed)
        c1, c2 = st.columns(2)
        c1.metric('Current Price', f'${current_price:.2f}')
        c2.metric('Predicted Next Close', f'${pred_price:.2f}', delta=f'{pct_change:.2f}%')

        # Create chart
        df['MA_short'] = df['close'].rolling(window=ma_short).mean()
        df['MA_long'] = df['close'].rolling(window=ma_long).mean()

        fig = go.Figure(data=[
            go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
        ])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA_short'], mode='lines', name=f'MA {ma_short}', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA_long'], mode='lines', name=f'MA {ma_long}', line=dict(color='magenta')))

        # mark prediction
        next_time = df['timestamp'].iloc[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(x=[next_time], y=[pred_price], mode='markers+text', text=[f'${pred_price:.2f}'], textposition='top center', marker=dict(color='gold', size=12), name='Predicted'))

        fig.update_layout(title=f'{ticker} Market Trend', template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # show table and export
        st.markdown('### Recent data (last 60 days)')
        st.dataframe(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(60).reset_index(drop=True))

        out = pd.DataFrame({
            'ticker': [ticker],
            'predicted_close': [pred_price],
            'predicted_change_pct': [float(pct_change)],
            'timestamp_utc': [datetime.utcnow()]
        })
        csv = out.to_csv(index=False).encode('utf-8')
        st.download_button('Download prediction (CSV)', csv, file_name=f'{ticker}_prediction.csv', mime='text/csv')

        st.success('Prediction complete ✅')

    else:
        st.info('👈 Enter inputs in the sidebar and click "Generate AI Prediction"')

if __name__ == '__main__':
    main()