import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import numpy as np

# 1. Page Setup
st.set_page_config(layout="wide", page_title="Ashwath's Pro Terminal")
st.title("Algorithmic Dashboard")

# --- UTILITY FUNCTIONS ---
def search_tickers(query):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        search_results = {}
        if 'quotes' in data:
            for quote in data['quotes']:
                if 'shortname' in quote and 'symbol' in quote:
                    label = f"{quote['symbol']} - {quote['shortname']}"
                    search_results[label] = quote['symbol']
        return search_results
    except:
        return {}

def is_support(df, i, n):
    for k in range(1, n+1):
        if i-k < 0 or i+k >= len(df): continue
        if df['Low'][i] >= df['Low'][i-k] or df['Low'][i] >= df['Low'][i+k]:
            return False
    return True

def is_resistance(df, i, n):
    for k in range(1, n+1): 
        if i-k < 0 or i+k >= len(df): continue
        if df['High'][i] <= df['High'][i-k] or df['High'][i] <= df['High'][i+k]:
            return False
    return True

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- NEW: AUTO-CALCULATOR FUNCTION ---
def count_levels(df, n, current_price):
    # This runs the logic silently to count lines
    levels = []
    # Optimization: Only scan every 2nd candle to speed up auto-calc
    for i in range(n, df.shape[0]-n):
        if is_support(df, i, n):
            l = df['Low'][i]
            if np.sum([abs(l - x) < (current_price*0.02) for x in levels]) == 0:
                levels.append(l)
        elif is_resistance(df, i, n):
            l = df['High'][i]
            if np.sum([abs(l - x) < (current_price*0.02) for x in levels]) == 0:
                levels.append(l)
    return len(levels)

# 2. Sidebar - Inputs
st.sidebar.header("Controls")
search_query = st.sidebar.text_input("Search Company", "Tata")

if search_query:
    results = search_tickers(search_query)
    ticker = results[list(results.keys())[0]] if results else None
    if results:
        selected_label = st.sidebar.selectbox("Select Stock", options=results.keys())
        ticker = results[selected_label]
    else:
        st.sidebar.error("No stocks found.")
else:
    ticker = None

period = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

# 3. Main Logic (Data Fetching FIRST)
# ... (keep your existing imports and setup)

# Add this NEW function near your other functions (like calculate_rsi)
@st.cache_data(ttl=300)  # 👈 This creates a 5-minute memory (Time To Live)
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    # yfinance sometimes fails, so we retry once if empty
    df = stock.history(period=period)
    return df

# ... (keep your sidebar code) ...

# 3. Main Dashboard Logic
if ticker:
    # CHANGE THIS PART to use the new cached function
    df = get_stock_data(ticker, period) 
    
    if not df.empty:
        # ... (rest of your code stays exactly the same)
        df = df[df['Volume'] > 0]
        current_price = df['Close'].iloc[-1]
        
        # --- SMART RECOMMENDATION ENGINE ---
        # We scan sensitivity (n) from 5 to 40 to find where we get 3-6 lines
        valid_n = []
        # Run a quick scan (step 3 to save performance)
        for n_scan in range(5, 45, 2): 
            count = count_levels(df, n_scan, current_price)
            if 3 <= count <= 6:
                valid_n.append(n_scan)
        
        if valid_n:
            rec_msg = f"💡 Rec. Range: {min(valid_n)} - {max(valid_n)}"
            rec_color = "green"
        else:
            rec_msg = "⚠️ Market is noisy. Try > 30"
            rec_color = "orange"

        # --- SIDEBAR SETTINGS (Rendered AFTER data check) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Technical Settings")
        
        st.sidebar.caption("Chart Overlays")
        show_price = st.sidebar.checkbox("Show Price", value=True)
        show_sma = st.sidebar.checkbox("Show SMA", value=True)
        show_support = st.sidebar.checkbox("Show Support (Grn)", value=True)
        show_resistance = st.sidebar.checkbox("Show Resistance (Red)", value=True)

        # The Recommendation Message
        st.sidebar.markdown(f":{rec_color}[{rec_msg}]")
        
        sensitivity = st.sidebar.number_input(
            "Sensitivity (Window Size)", 
            min_value=2, 
            max_value=50, 
            value=10, 
            step=1
        )

        # --- DATA PROCESSING & PLOTTING ---
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        prev_close = df['Close'].iloc[-2]
        day_change = ((current_price - prev_close) / prev_close) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"₹{current_price:.2f}", f"{day_change:.2f}%")
        col2.metric("Day High", f"₹{df['High'].max():.2f}")
        col3.metric("Day Low", f"₹{df['Low'].min():.2f}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # 1. Price
        if show_price:
            fig.add_trace(go.Candlestick(x=df.index,
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'],
                            name="Price"), row=1, col=1)
        # 2. SMA
        if show_sma:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', 
                                     name='50-Day SMA', line=dict(color='orange')), row=1, col=1)
        
        # 3. Levels
        levels = []
        if show_support or show_resistance:
            n = int(sensitivity)
            for i in range(n, df.shape[0]-n):
                if show_support and is_support(df, i, n):
                    l = df['Low'][i]
                    if np.sum([abs(l - x[1]) < (current_price*0.02) for x in levels]) == 0:
                        levels.append((df.index[i], l, "Support"))
                elif show_resistance and is_resistance(df, i, n):
                    l = df['High'][i]
                    if np.sum([abs(l - x[1]) < (current_price*0.02) for x in levels]) == 0:
                        levels.append((df.index[i], l, "Resistance"))

            for date, level, kind in levels:
                color = "green" if kind == "Support" else "red"
                fig.add_hline(y=level, line_dash="dot", line_color=color, row=1, col=1)

        # 4. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', 
                                 line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(title=f"{ticker} Analysis", height=700, 
                          xaxis_rangeslider_visible=False,
                          legend=dict(x=1.02, y=1))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Error loading data.")