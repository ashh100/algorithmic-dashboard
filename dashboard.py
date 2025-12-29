import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import xml.etree.ElementTree as ET

# 1. Page Setup
st.set_page_config(layout="wide", page_title="Ashwath's Pro Terminal")
st.title("Algorithmic Dashboard")

# --- SESSION STATE SETUP ---
if 'show_ai' not in st.session_state:
    st.session_state.show_ai = False

# Initialize Portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# Initialize Ticker
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "RELIANCE.NS"

# Initialize Search Key Wrapper
if "search_key_trigger" not in st.session_state:
    st.session_state.search_key_trigger = 0

def run_ai_analysis():
    st.session_state.show_ai = True

def go_back():
    st.session_state.show_ai = False

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
                symbol = quote.get('symbol', '')
                if symbol.endswith('.NS') or symbol.endswith('.BO'):
                    shortname = quote.get('shortname', symbol)
                    label = f"{symbol} - {shortname}"
                    search_results[label] = symbol
        return search_results
    except:
        return {}

def is_support(df, i, n):
    try:
        # Using .iloc ensures we look at the row position, not the Date index
        for k in range(1, n+1):
            if i-k < 0 or i+k >= len(df): 
                return False
            if df['Low'].iloc[i] >= df['Low'].iloc[i-k] or df['Low'].iloc[i] >= df['Low'].iloc[i+k]:
                return False
        return True
    except Exception:
        return False

def is_resistance(df, i, n):
    try:
        for k in range(1, n+1): 
            if i-k < 0 or i+k >= len(df): 
                return False
            if df['High'].iloc[i] <= df['High'].iloc[i-k] or df['High'].iloc[i] <= df['High'].iloc[i+k]:
                return False
        return True
    except Exception:
        return False
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- PATTERN RECOGNITION ---
def detect_candlestick_patterns(df):
    patterns = []
    offset_pct = 0.015 
    
    # --- OPTION A: TA-LIB (Try/Except) ---
    try:
        import talib
        hammer = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        star = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        engulfing = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        morning = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        evening = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        for i in range(len(df)):
            if df['Volume'].iloc[i] == 0: continue
            date = df.index[i]
            
            if morning[i] == 100:
                patterns.append({'date': date, 'price': df['Low'].iloc[i] * (1 - offset_pct), 'label': 'Morn. Star', 'color': 'green', 'symbol': 'triangle-up'})
            elif evening[i] == -100:
                patterns.append({'date': date, 'price': df['High'].iloc[i] * (1 + offset_pct), 'label': 'Even. Star', 'color': 'red', 'symbol': 'triangle-down'})
            elif engulfing[i] == 100:
                patterns.append({'date': date, 'price': df['Low'].iloc[i] * (1 - offset_pct), 'label': 'Bull Engulf', 'color': 'cyan', 'symbol': 'triangle-up'})
            elif engulfing[i] == -100:
                patterns.append({'date': date, 'price': df['High'].iloc[i] * (1 + offset_pct), 'label': 'Bear Engulf', 'color': 'magenta', 'symbol': 'triangle-down'})
            elif hammer[i] == 100:
                patterns.append({'date': date, 'price': df['Low'].iloc[i] * (1 - offset_pct), 'label': 'Hammer', 'color': 'yellow', 'symbol': 'triangle-up'})
            elif star[i] == -100:
                patterns.append({'date': date, 'price': df['High'].iloc[i] * (1 + offset_pct), 'label': 'Shoot. Star', 'color': 'orange', 'symbol': 'triangle-down'})
                
        return patterns

    # --- OPTION B: MANUAL MATH ---
    except ImportError:
        df = df.copy() 
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['Color'] = np.where(df['Close'] >= df['Open'], 'Green', 'Red')
        
        for i in range(1, len(df)):
            date = df.index[i]
            body = df['Body'].iloc[i]
            upper = df['Upper_Shadow'].iloc[i]
            lower = df['Lower_Shadow'].iloc[i]
            color = df['Color'].iloc[i]
            
            prev_open = df['Open'].iloc[i-1]
            prev_close = df['Close'].iloc[i-1]
            prev_color = df['Color'].iloc[i-1]
            
            if (lower > 2 * body) and (upper < 0.5 * body) and (body > 0):
                patterns.append({'date': date, 'price': df['Low'].iloc[i] * (1 - offset_pct), 'label': 'Hammer', 'color': 'yellow', 'symbol': 'triangle-up'})
            elif (upper > 2 * body) and (lower < 0.5 * body) and (body > 0):
                 patterns.append({'date': date, 'price': df['High'].iloc[i] * (1 + offset_pct), 'label': 'Shoot. Star', 'color': 'orange', 'symbol': 'triangle-down'})
            elif (color == 'Green') and (prev_color == 'Red'):
                if (df['Close'].iloc[i] > prev_open) and (df['Open'].iloc[i] < prev_close):
                       patterns.append({'date': date, 'price': df['Low'].iloc[i] * (1 - offset_pct), 'label': 'Bull Engulf', 'color': 'cyan', 'symbol': 'triangle-up'})
            elif (color == 'Red') and (prev_color == 'Green'):
                if (df['Close'].iloc[i] < prev_open) and (df['Open'].iloc[i] > prev_close):
                    patterns.append({'date': date, 'price': df['High'].iloc[i] * (1 + offset_pct), 'label': 'Bear Engulf', 'color': 'magenta', 'symbol': 'triangle-down'})

        return patterns

# --- AUTO-SENSITIVITY ---
def calculate_optimal_sensitivity(df):
    test_values = [20, 15, 25, 30, 10, 35, 40]
    best_n = 20
    min_dist_to_ideal = float('inf')
    current_price = df['Close'].iloc[-1]
    
    for n in test_values:
        level_count = 0
        levels = []
        for i in range(n, df.shape[0]-n):
            if is_support(df, i, n):
                l = df['Low'][i]
                if np.sum([abs(l - x) < (current_price*0.02) for x in levels]) == 0:
                    levels.append(l)
                    level_count += 1
            elif is_resistance(df, i, n):
                l = df['High'][i]
                if np.sum([abs(l - x) < (current_price*0.02) for x in levels]) == 0:
                    levels.append(l)
                    level_count += 1
        if 3 <= level_count <= 5: return n 
        dist = abs(level_count - 4)
        if dist < min_dist_to_ideal:
            min_dist_to_ideal = dist
            best_n = n
    return best_n

# --- FUNDAMENTALS ---
@st.cache_data(ttl=86400) 
def get_company_info(ticker):
    stock = yf.Ticker(ticker)
    info = None
    try: 
        info = stock.info
    except Exception: 
        pass 
    
    if not info: info = {}

    # If yfinance failed to get fundamental data, use Alpha Vantage Fallback
    if 'marketCap' not in info or info.get('sector') == "N/A":
        try:
            import requests
            # Accessing the key you put in .streamlit/secrets.toml
            av_key = st.secrets["ALPHA_VANTAGE_KEY"]
            clean_symbol = ticker.replace(".NS", "").replace(".BO", "")
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={clean_symbol}&apikey={av_key}"
            res = requests.get(url, timeout=5).json()
            
            if res and "Sector" in res:
                info['sector'] = res.get("Sector", "N/A")
                info['marketCap'] = float(res.get("MarketCapitalization", 0))
                info['trailingPE'] = float(res.get("PERatio", 0)) if res.get("PERatio") != "None" else "N/A"
                info['dividendYield'] = float(res.get("DividendYield", 0))
        except Exception:
            pass

    # Your original fast_info logic as a secondary backup
    if 'marketCap' not in info:
        try:
            fast = stock.fast_info
            if fast and fast.market_cap:
                info['marketCap'] = fast.market_cap
        except Exception: 
            pass

    # Fill missing keys to prevent UI KeyErrors
    if 'sector' not in info: info['sector'] = "N/A"
    if 'trailingPE' not in info: info['trailingPE'] = "N/A"
    if 'dividendYield' not in info: info['dividendYield'] = 0
    if 'marketCap' not in info: info['marketCap'] = 0

    return info

# --- COMPARISON ---
@st.cache_data(ttl=3600)
def get_market_comparison(ticker, period):
    try:
        tickers = [ticker, "^NSEI"]
        data = yf.download(tickers, period=period)['Close']
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        normalized = (data / data.iloc[0] - 1) * 100
        correlation = data.corr().iloc[0, 1]
        return normalized, correlation
    except Exception as e:
        return None, 0

# --- NEWS ---
@st.cache_data(ttl=3600)
def get_stock_news(ticker):
    try:
        search_term = ticker.split('.')[0] 
        url = f"https://news.google.com/rss/search?q={search_term}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall('.//item')[:5]:
            title = item.find('title').text
            link = item.find('link').text
            source = item.find('source')
            publisher = source.text if source is not None else "Google News"
            news_items.append({'title': title, 'link': link, 'publisher': publisher})
        return news_items
    except Exception as e:
        return []

# --- AI ANALYST ---
def get_ai_analysis(ticker, data, news_list):
    recent_data = data.tail(10).to_string()
    current_price = data['Close'].iloc[-1]
    
    if news_list:
        formatted_news = []
        for item in news_list:
            if isinstance(item, dict):
                formatted_news.append(f"- {item['title']} (Source: {item['publisher']})")
            else:
                formatted_news.append(f"- {item}")
        news_context = "\n".join(formatted_news)
    else:
        news_context = "No recent news available."
    
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
        else:
            return "⚠️ Error: GROQ_API_KEY missing in .streamlit/secrets.toml"
    except Exception:
        return "⚠️ Error: Could not load secrets."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"""
    You are a Wall Street analyst. Analyze {ticker} based on this data:
    Price: {current_price}
    Recent Data: {recent_data}
    News: {news_context}
    
    Provide a "Trader's Take":
    1. Trend (Bullish/Bearish/Neutral)
    2. One Key Reason
    3. Action (Buy/Sell/Wait)
    """
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
             return f"API Error {response.status_code}: {response.text}"
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Connection Error: {str(e)}"

# --- DATA FETCHING ---
@st.cache_data(ttl=300) 
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    
    # --- PART 1: PRICE HISTORY (Kept Original) ---
    df = stock.history(period=period)
    if df.empty: 
        return df, {"sector": "N/A", "pe": "N/A", "mcap": "N/A"}
    
    # --- PART 2: FUNDAMENTALS (New Safe Fallback) ---
    fundamentals = {"sector": "N/A", "pe": "N/A", "mcap": "N/A"}
    try:
        # Try yfinance first (using .info)
        info = stock.info
        fundamentals["sector"] = info.get('sector', 'N/A')
        fundamentals["pe"] = info.get('trailingPE', 'N/A')
        fundamentals["mcap"] = info.get('marketCap', 'N/A')
        
        # If yfinance failed to get a sector, try Alpha Vantage secret
        if fundamentals["sector"] == "N/A" and "ALPHA_VANTAGE_KEY" in st.secrets:
            import requests
            clean_symbol = ticker.replace(".NS", "").replace(".BO", "")
            av_key = st.secrets["ALPHA_VANTAGE_KEY"]
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={clean_symbol}&apikey={av_key}"
            res = requests.get(url, timeout=5).json()
            if "Sector" in res:
                fundamentals["sector"] = res.get("Sector", "N/A")
                fundamentals["pe"] = res.get("PERatio", "N/A")
                mcap_raw = float(res.get("MarketCapitalization", 0))
                fundamentals["mcap"] = mcap_raw if mcap_raw > 0 else "N/A"
    except Exception:
        pass # If everything fails, we still have "N/A" defaults

    # --- PART 3: TECHNICALS (Kept Original Variable Names) ---
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Pct_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Pct_Change'].rolling(window=20).std()
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)
    
    # IMPORTANT: Returning both the DF and the Fundamentals dict
    return df, fundamentals

# --- SIDEBAR & CONTROLS ---
st.sidebar.header("Controls")

search_key = f"sidebar_search_{st.session_state.search_key_trigger}"
search_query = st.sidebar.text_input("Search Company", key=search_key)

if search_query:
    results = search_tickers(search_query)
    if results:
        selected_label = st.sidebar.selectbox("Select Stock", options=results.keys())
        ticker = results[selected_label]
        st.session_state["ticker"] = ticker
    else:
        st.sidebar.error("No stocks found.")
        ticker = None
else:
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "RELIANCE.NS"
    ticker = st.session_state["ticker"]

period = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

if ticker:
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📍 {ticker}") 
    
    # This calls your function which now includes the Alpha Vantage fallback
    info = get_company_info(ticker)
    
    # Initialize defaults to prevent crashes if info is None or empty
    sector = "N/A"
    pe_ratio = "N/A"
    market_cap = 0
    div_yield = 0.0

    if info:
        sector = info.get('sector', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', 0)
        
        # Safe dividend yield calculation
        dy = info.get('dividendYield')
        div_yield = (dy * 100) if isinstance(dy, (int, float)) else 0.0
    
    # --- SAFE P/E FORMATTING ---
    if isinstance(pe_ratio, (int, float)):
        pe_str = f"{pe_ratio:.2f}"
    else:
        pe_str = "N/A"

    # --- SAFE MARKET CAP FORMATTING (Fixes the TypeError) ---
    # We check if market_cap is a number before doing math
    if isinstance(market_cap, (int, float)) and market_cap > 0:
        if market_cap >= 1e12: 
            mcap_str = f"₹{market_cap/1e12:.2f}T"
        elif market_cap >= 1e7: 
            # Added Crore (Cr) formatting which is more standard for NSE
            mcap_str = f"₹{market_cap/1e7:.2f} Cr"
        elif market_cap >= 1e5:
            mcap_str = f"₹{market_cap/1e5:.2f} L"
        else:
            mcap_str = f"₹{market_cap:,.0f}"
    else:
        mcap_str = "N/A"

    # --- UI DISPLAY ---
    st.sidebar.info(f"**Sector:** {sector}")
    st.sidebar.metric("Market Cap", mcap_str)
    st.sidebar.metric("P/E Ratio", pe_str)
    st.sidebar.metric("Div Yield", f"{div_yield:.2f}%")
else:
    st.sidebar.warning("Fundamental data not available")
st.sidebar.markdown("---")
st.sidebar.subheader("Chart Display")
chart_type = st.sidebar.selectbox("Chart Style", ["Candlestick", "Line", "Area", "OHLC"])

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Analyst")
st.sidebar.button("Generate Report", on_click=run_ai_analysis)

# --- MAIN DASHBOARD LOGIC ---
if ticker:
    df, info_data = get_stock_data(ticker, period)
    
    if not df.empty:
        if not df[df['Volume'] > 0].empty:
            df = df[df['Volume'] > 0]
        
        current_price = df['Close'].iloc[-1]
        
        period_high = df['High'].max()
        period_low = df['Low'].min()
        high_date = df['High'].idxmax()
        low_date = df['Low'].idxmin()

        optimal_n = calculate_optimal_sensitivity(df)

        st.sidebar.caption("Overlays")
        show_ema = st.sidebar.checkbox("Show EMA (20/50)", value=True)
        show_support = st.sidebar.checkbox("Show Support", value=True)
        show_resistance = st.sidebar.checkbox("Show Resistance", value=True)
        show_patterns = st.sidebar.checkbox("Show Patterns (Hammer/Engulfing)", value=False)
        show_bb = st.sidebar.checkbox('Show Bollinger Bands')
        
        sensitivity = st.sidebar.number_input("Sensitivity (Auto-Optimized)", 
                                              min_value=2, max_value=50, 
                                              value=optimal_n, 
                                              key=f"sens_{ticker}")
        
        df['RSI'] = calculate_rsi(df['Close'])
        current_volatility = df['Volatility'].iloc[-1] * 100 if not np.isnan(df['Volatility'].iloc[-1]) else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"₹{current_price:.2f}", f"{((current_price - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100:.2f}%")
        col2.metric("High", f"₹{period_high:.2f}")
        col3.metric("Low", f"₹{period_low:.2f}")
        col4.metric("Volatility", f"{current_volatility:.2f}%")

        if st.session_state.show_ai:
            if st.button("← Back to Dashboard", type="primary"):
                go_back()
                st.rerun()

            with st.spinner(f"Reading news & analyzing charts for {ticker}..."):
                news_headlines = get_stock_news(ticker)
                analysis = get_ai_analysis(ticker, df, news_headlines)
                st.info(f"**AI Analysis:**\n\n{analysis}")
                
                with st.expander("📰 Read the News Headlines Used by AI"):
                    if not news_headlines:
                        st.write("No specific news found.")
                    else:
                        for i, n in enumerate(news_headlines):
                            if isinstance(n, dict) and 'title' in n and 'link' in n:
                                st.markdown(f"### {i+1}. [{n['title']}]({n['link']})")
                                st.caption(f"Published by: {n.get('publisher', 'Unknown')}")
                                st.divider()
                            else:
                                st.write(n)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Action", "📊 Technical Indicators", "⚖️ Sector Comparison", "💼 Portfolio"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=df.index, 
                    open=df['Open'], 
                    high=df['High'], 
                    low=df['Low'], 
                    close=df['Close'], 
                    name="Price",
                    increasing_line_color='#00e676', 
                    decreasing_line_color='#ff1744', 
                    increasing_line_width=1.5, 
                    decreasing_line_width=1.5
                ), row=1, col=1)
            elif chart_type == "Line":
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close", line=dict(color='#00e676')), row=1, col=1)
            elif chart_type == "Area":
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tozeroy', mode='lines', name="Close", line=dict(color='#2962ff')), row=1, col=1)
            elif chart_type == "OHLC":
                fig.add_trace(go.Ohlc(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)

            if show_ema:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#ff9100', width=1), name='EMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#2962ff', width=1), name='EMA 50'), row=1, col=1)

            if show_bb and 'Upper_Band' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='rgba(255, 255, 255, 0.1)'), name='Upper Band'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='rgba(255, 255, 255, 0.1)'), name='Lower Band', fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)'), row=1, col=1)

            if show_patterns:
                patterns = detect_candlestick_patterns(df)
                if patterns:
                    for p in patterns:
                        fig.add_trace(go.Scatter(
                            x=[p['date']], y=[p['price']],
                            mode='markers',
                            marker=dict(symbol=p['symbol'], size=12, color=p['color']),
                            name=p['label'],
                            hovertext=p['label']
                        ), row=1, col=1)
                else:
                    st.toast("No major patterns found in this period.")

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

            colors = ['#00e676' if r['Open'] - r['Close'] <= 0 else '#ff1744' for i, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Momentum & Strength")
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5])
            
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#b388ff', width=2), name='RSI'), row=1, col=1)
            fig_macd.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig_macd.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            fig_macd.update_yaxes(title_text="RSI", row=1, col=1)
            
            if 'MACD' in df.columns:
                colors_macd = ['#00e676' if val >= 0 else '#ff1744' for val in df['MACD'] - df['Signal_Line']]
                fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal_Line'], marker_color=colors_macd, name='MACD Hist'), row=2, col=1)
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2962ff', width=1), name='MACD'), row=2, col=1)
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='#ff9100', width=1), name='Signal'), row=2, col=1)
                fig_macd.update_yaxes(title_text="MACD", row=2, col=1)

            fig_macd.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig_macd, use_container_width=True)

        with tab3:
            st.subheader(f"Performance: {ticker} vs Nifty 50")
            comp_data, correlation = get_market_comparison(ticker, period)
            
            if comp_data is not None and not comp_data.empty:
                c1, c2 = st.columns([1, 3])
                c1.metric("Correlation", f"{correlation:.2f}")
                if correlation > 0.7: c1.success("Moves with Market")
                elif correlation < 0.3: c1.warning("Moves Independently")
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=comp_data.index, y=comp_data.iloc[:, 0], name=ticker, line=dict(color='#00e676', width=2)))
                fig_comp.add_trace(go.Scatter(x=comp_data.index, y=comp_data.iloc[:, 1], name='Nifty 50', line=dict(color='white', width=1, dash='dash')))
                
                fig_comp.update_layout(title="Relative Return (%)", template="plotly_dark", height=500, yaxis_title="% Change", margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Could not fetch comparison data. (Nifty 50 data may be unavailable).")
        
        with tab4:
            st.subheader("💼 Portfolio Tracker")
            
            with st.expander("Add New Position", expanded=False):
                st.caption("Step 1: Search & Select Stock")
                
                row1_1, row1_2 = st.columns(2)
                
                with row1_1:
                    pf_search = st.text_input("Search Ticker", placeholder="e.g. TCS, HDFC, Tata", key="pf_search_box")
                
                chosen_ticker = None
                
                with row1_2:
                    if pf_search:
                        results = search_tickers(pf_search)
                        if results:
                            selected_label = st.selectbox("Select Stock", options=results.keys(), key="pf_select_box")
                            chosen_ticker = results[selected_label]
                        else:
                            st.warning("No Indian stocks found.")
                            
                if chosen_ticker:
                    if 'last_pf_ticker' not in st.session_state:
                        st.session_state.last_pf_ticker = None
                    
                    if st.session_state.last_pf_ticker != chosen_ticker:
                        try:
                            stock_info = yf.Ticker(chosen_ticker)
                            try:
                                default_price = stock_info.fast_info.last_price
                            except:
                                default_price = stock_info.history(period='1d')['Close'].iloc[-1]
                        except:
                            default_price = 0.0
                        
                        st.session_state.pf_ticker_input = chosen_ticker
                        st.session_state.pf_price_input = float(default_price)
                        st.session_state.last_pf_ticker = chosen_ticker

                st.divider()
                st.caption("Step 2: Confirm Details")
                
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                
                with c1:
                    new_ticker = st.text_input("Ticker Symbol", key="pf_ticker_input").upper()
                with c2:
                    new_qty = st.number_input("Quantity", min_value=1, value=10, key="pf_qty_input")
                with c3:
                    new_price = st.number_input("Avg Buy Price", min_value=0.0, format="%.2f", key="pf_price_input")
                with c4:
                    st.write("") 
                    st.write("")
                    # --- FIXED SECTION START ---
                    # We ONLY update the portfolio list, NOT the global dashboard ticker.
                    if st.button("Add"):
                        if new_ticker and new_price > 0:
                            st.session_state.portfolio.append({
                                "Ticker": new_ticker,
                                "Quantity": new_qty,
                                "Buy Price": new_price
                            })
                            st.toast(f"Added {new_ticker} to Portfolio!", icon="✅")
                            # No st.rerun() here.
                            # No st.session_state['ticker'] update here.
                        else:
                            st.error("Invalid Ticker")
                    # --- FIXED SECTION END ---

            if st.session_state.portfolio:
                pf_df = pd.DataFrame(st.session_state.portfolio)
                pf_df['Ticker'] = pf_df['Ticker'].astype(str).str.strip().str.upper()

                realtime_prices = {}
                unique_tickers = pf_df['Ticker'].unique()

                for t in unique_tickers:
                    try:
                        stock = yf.Ticker(t)
                        try:
                            current_price = stock.fast_info.last_price
                        except:
                            hist = stock.history(period="1d")
                            if not hist.empty:
                                current_price = hist['Close'].iloc[-1]
                            else:
                                current_price = 0.0
                        realtime_prices[t] = float(current_price)
                    except Exception as e:
                        realtime_prices[t] = 0.0

                pf_df['Current Price'] = pf_df['Ticker'].map(realtime_prices).fillna(0.0)
                pf_df['Invested Value'] = pf_df['Quantity'] * pf_df['Buy Price']
                pf_df['Current Value'] = pf_df['Quantity'] * pf_df['Current Price']
                pf_df['P/L'] = pf_df['Current Value'] - pf_df['Invested Value']
                pf_df['P/L %'] = (pf_df['P/L'] / pf_df['Invested Value']) * 100

                total_invested = pf_df['Invested Value'].sum()
                total_current = pf_df['Current Value'].sum()
                total_pl = pf_df['P/L'].sum()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Invested", f"₹{total_invested:,.2f}")
                m2.metric("Current Value", f"₹{total_current:,.2f}")
                m3.metric("Total P/L", f"₹{total_pl:,.2f}", delta=f"{total_pl:,.2f}")

                st.dataframe(pf_df.style.format({
                    "Buy Price": "₹{:.2f}", 
                    "Current Price": "₹{:.2f}",
                    "Invested Value": "₹{:.2f}",
                    "Current Value": "₹{:.2f}",
                    "P/L": "₹{:.2f}",
                    "P/L %": "{:.2f}%"
                }).applymap(lambda x: 'color: #00e676' if x > 0 else 'color: #ff1744', subset=['P/L', 'P/L %']),
                use_container_width=True)
                
                if st.button("Clear Portfolio", type="secondary"):
                    st.session_state.portfolio = []
                    st.rerun()
            else:
                st.info("Portfolio is empty.")
else:
    st.error("Error loading data.")