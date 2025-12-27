import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import numpy as np
import time

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

def count_levels(df, n, current_price):
    levels = []
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

# --- NEWS FUNCTION ---
@st.cache_data(ttl=3600)
def get_stock_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        headlines = []
        if news:
            for n in news[:5]: 
                headlines.append(f"- {n['title']}")
        return headlines
    except:
        return []

# --- NEW: HUGGING FACE AI INTEGRATION ---
# --- NEW: HUGGING FACE AI INTEGRATION (FIXED & STABLE) ---
def get_ai_analysis(ticker, data, news_list):
    """Calls Hugging Face Inference API for analysis."""
    
    # 1. Prepare Data
    recent_data = data.tail(10).to_string()
    current_price = data['Close'].iloc[-1]
    news_context = "\n".join(news_list) if news_list else "No recent news available."
    
    # 2. Define the Model (Switching to v0.2 which is NOT gated)
    # This URL is the standard, stable free tier endpoint.
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    # 3. Get Token safely
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except:
        return "⚠️ Error: HF_TOKEN missing in .streamlit/secrets.toml"

    headers = {"Authorization": f"Bearer {hf_token}"}

    # 4. Prompt Engineering
    prompt = f"""
    [INST] You are a financial analyst. Analyze {ticker} based on this data:
    
    Current Price: {current_price}
    Recent Data:
    {recent_data}
    
    News:
    {news_context}
    
    Give a 3-sentence summary: Trend (Bullish/Bearish), Key Reason, and Action (Buy/Sell/Wait).
    [/INST]
    """
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    # 5. Call API with BETTER Error Handling
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # If the API returns an error (like 404 or 503), show the text
        if response.status_code != 200:
             return f"API Error {response.status_code}: {response.text}"

        result = response.json()
        
        # Handle list vs dict response
        if isinstance(result, list) and len(result) > 0:
             return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
             return result['generated_text']
        else:
             return f"Unexpected format: {result}"
        
    except Exception as e:
        return f"Connection Error: {str(e)}"

# --- DATA FETCHING ---
@st.cache_data(ttl=300) 
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty: return df

    # EMA & Volatility
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Pct_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Pct_Change'].rolling(window=20).std()
    
    return df

@st.cache_data(ttl=86400) 
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except:
        return None

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

# --- FUNDAMENTALS SECTION ---
if ticker:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 Fundamentals")
    info = get_company_info(ticker)
    
    if info:
        sector = info.get('sector', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', 0)
        div_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        if market_cap > 1e12: mcap_str = f"₹{market_cap/1e12:.2f}T"
        elif market_cap > 1e9: mcap_str = f"₹{market_cap/1e9:.2f}B"
        else: mcap_str = f"₹{market_cap/1e6:.2f}M"

        st.sidebar.info(f"**Sector:** {sector}")
        st.sidebar.metric("Market Cap", mcap_str)
        st.sidebar.metric("P/E Ratio", f"{pe_ratio}")
        st.sidebar.metric("Div Yield", f"{div_yield:.2f}%")
    else:
        st.sidebar.warning("Fundamental data not available")

# --- CHART STYLE SELECTOR ---
st.sidebar.markdown("---")
st.sidebar.subheader("Chart Display")
chart_type = st.sidebar.selectbox("Chart Style", ["Candlestick", "Line", "Area", "OHLC"])

# --- AI ANALYST BUTTON ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Analyst")
if st.sidebar.button("Generate Report"):
    run_ai = True
else:
    run_ai = False

# 3. Main Dashboard Logic
if ticker:
    df = get_stock_data(ticker, period) 
    
    if not df.empty:
        # Safety check for volume
        if not df[df['Volume'] > 0].empty:
            df = df[df['Volume'] > 0]
        
        current_price = df['Close'].iloc[-1]
        
        # --- AUTO SENSITIVITY ENGINE ---
        valid_n = []
        for n_scan in range(5, 45, 2): 
            count = count_levels(df, n_scan, current_price)
            if 3 <= count <= 6: valid_n.append(n_scan)
        
        if valid_n:
            optimal_n = int(sum(valid_n) / len(valid_n))
        else:
            optimal_n = 10

        # Sidebar Settings
        st.sidebar.caption("Overlays")
        show_ema = st.sidebar.checkbox("Show EMA (20/50)", value=True)
        show_support = st.sidebar.checkbox("Show Support", value=True)
        show_resistance = st.sidebar.checkbox("Show Resistance", value=True)
        sensitivity = st.sidebar.number_input("Sensitivity", 2, 50, optimal_n)

        # Metrics
        df['RSI'] = calculate_rsi(df['Close'])
        current_volatility = df['Volatility'].iloc[-1] * 100 if not np.isnan(df['Volatility'].iloc[-1]) else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"₹{current_price:.2f}", f"{((current_price - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100:.2f}%")
        col2.metric("High", f"₹{df['High'].max():.2f}")
        col3.metric("Low", f"₹{df['Low'].min():.2f}")
        col4.metric("Volatility", f"{current_volatility:.2f}%")

        # --- AI REPORT SECTION ---
        if run_ai:
            with st.spinner(f"Reading news & analyzing charts for {ticker}..."):
                news_headlines = get_stock_news(ticker)
                analysis = get_ai_analysis(ticker, df, news_headlines)
                st.info(f"**AI Analysis:**\n\n{analysis}")
                
                with st.expander("📰 Read the News Headlines Used by AI"):
                    if news_headlines:
                        for h in news_headlines:
                            st.write(h)
                    else:
                        st.write("No specific news found.")

        # --- PLOTTING LOGIC ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.6, 0.2, 0.2], 
                            subplot_titles=("Price Action", "Volume", "RSI"))

        # 1. Main Chart
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                         low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                                     name="Close", line=dict(color='#00e676')), row=1, col=1)
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tozeroy', mode='lines', 
                                     name="Close", line=dict(color='#2962ff')), row=1, col=1)
        elif chart_type == "OHLC":
            fig.add_trace(go.Ohlc(x=df.index, open=df['Open'], high=df['High'], 
                                  low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)

        # 2. Overlays
        if show_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#00e676', width=1), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#2962ff', width=1), name='EMA 50'), row=1, col=1)
        
        # 3. Support/Resistance
        if show_support or show_resistance:
            levels = []
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
                fig.add_hline(y=level, line_dash="dot", line_color=color, row=1, col=1, opacity=0.5)

        # 4. Volume & RSI
        colors = ['#00e676' if r['Open'] - r['Close'] <= 0 else '#ff1744' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#b388ff'), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Error loading data.")