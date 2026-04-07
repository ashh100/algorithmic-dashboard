import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import xml.etree.ElementTree as ET
from nsepython import nse_eq
from nselib import capital_market
from bs4 import BeautifulSoup
import database as db
import data_engine
# Initialize DB (Runs once)
db.init_db()
st.cache_data.clear()

# --- BRIDGE FUNCTIONS ---

# 1. Search Tickers (Directly calls the engine)
def search_tickers(query):
    return data_engine.search_tickers(query)

# 2. Get Fundamentals (The "Wrapper" with Caching)
# We keep the name "get_nse_fundamentals" so the rest of your app doesn't break!
@st.cache_data(ttl=86400)
def get_nse_fundamentals(ticker):
    return data_engine.get_nse_fundamentals(ticker)
# 1. Page Setup
st.set_page_config(layout="wide", page_title="Ashwath's Pro Terminal", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* === HIDE STREAMLIT CHROME === */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Make header dark — do NOT hide it, the sidebar toggle lives inside it */
[data-testid="stHeader"] {
    background-color: #0b0e14 !important;
    border-bottom: 1px solid #1e2533 !important;
}
/* Hide only the deploy/share button, not the whole toolbar */
[data-testid="stToolbarActions"] { visibility: hidden; }

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0b0e14; }
::-webkit-scrollbar-thumb { background: #1e2533; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2962ff; }

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1e2533 !important;
}
[data-testid="stSidebar"] label {
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0d1117;
    border-bottom: 1px solid #1e2533;
    gap: 0px;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: #6b7280;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 10px 20px;
    border-radius: 0;
    transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #f8f9fa;
    background-color: #151a23;
}
.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    border-bottom: 2px solid #00e676 !important;
    color: #00e676 !important;
}

/* === METRIC CARDS === */
div[data-testid="metric-container"] {
    background-color: #0d1117;
    border: 1px solid #1e2533;
    border-top: 2px solid #2962ff;
    padding: 14px 18px;
    border-radius: 3px;
    transition: border-top-color 0.2s;
    box-shadow: none;
}
div[data-testid="metric-container"]:hover {
    border-top-color: #00e676;
    transform: none;
    box-shadow: none;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-size: 9px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #f8f9fa !important;
}

/* === BUTTONS === */
.stButton > button {
    background: transparent;
    border: 1px solid #1e2533;
    color: #9ca3af;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 2px;
    padding: 8px 20px;
    transition: all 0.15s;
    width: 100%;
}
.stButton > button:hover {
    border-color: #2962ff;
    color: #f8f9fa;
    background: rgba(41,98,255,0.08);
}
[data-testid="stButton"] button[kind="primary"] {
    background: #2962ff;
    border-color: #2962ff;
    color: white;
}

/* === INPUTS === */
.stTextInput input, .stNumberInput input {
    background-color: #0d1117 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 2px !important;
    color: #f8f9fa !important;
    font-size: 12px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #2962ff !important;
    box-shadow: none !important;
}
.stSelectbox > div > div {
    background-color: #0d1117 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 2px !important;
}

/* === EXPANDER === */
details summary {
    font-size: 11px !important;
    letter-spacing: 1px !important;
    color: #9ca3af !important;
}

/* === DIVIDER === */
hr { border-color: #1e2533 !important; }

/* === HEADER === */
.terminal-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding-bottom: 14px;
    border-bottom: 1px solid #1e2533;
    margin-bottom: 22px;
}
.terminal-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 5px;
    color: #f8f9fa;
    text-transform: uppercase;
}
.terminal-title span { color: #00e676; }
.terminal-badge {
    font-size: 9px;
    letter-spacing: 2px;
    color: #00e676;
    border: 1px solid #00e676;
    padding: 2px 8px;
    border-radius: 2px;
    margin-left: 14px;
    vertical-align: middle;
    opacity: 0.8;
}
.terminal-meta {
    font-size: 10px;
    color: #374151;
    letter-spacing: 1px;
}
.sidebar-brand {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 4px;
    color: #00e676;
    text-transform: uppercase;
    padding-bottom: 12px;
    border-bottom: 1px solid #1e2533;
    margin-bottom: 4px;
}
.section-label {
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #374151;
    border-left: 2px solid #2962ff;
    padding-left: 8px;
    margin: 16px 0 10px 0;
}
.ai-card {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-left: 3px solid #2962ff;
    padding: 20px 24px;
    border-radius: 3px;
    font-size: 13px;
    line-height: 1.7;
    color: #d1d5db;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

import datetime
now = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
st.markdown(f"""
<div class="terminal-header">
    <div>
        <span class="terminal-title">Market <span>Terminal</span></span>
        <span class="terminal-badge">LIVE</span>
    </div>
    <div class="terminal-meta">NSE · BSE &nbsp;|&nbsp; {now}</div>
</div>
""", unsafe_allow_html=True)

# --- SESSION STATE SETUP ---
if 'show_ai' not in st.session_state:
    st.session_state.show_ai = False

# Initialize Portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# Initialize Ticker
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "RELIANCE.NS"


def run_ai_analysis():
    st.session_state.show_ai = True

def go_back():
    st.session_state.show_ai = False

@st.cache_data
def get_all_stock_symbols():
    try:
        from nselib import capital_market
        # 1. Get Official List
        nse_df = capital_market.equity_list()
        all_symbols = nse_df['SYMBOL'].tolist()
    except:
        all_symbols = []

    # 2. MANUALLY ADD Missing/New Stocks (The Fix for Zomato)
    # Add any other missing tickers here
    missing_stocks = ["ZOMATO", "PAYTM", "JIOFIN", "SWIGGY", "LICI"] 
    
    for stock in missing_stocks:
        if stock not in all_symbols:
            all_symbols.append(stock)
            
    # 3. Sort list for easy searching
    return sorted(all_symbols)

# --- HOW TO USE IN SIDEBAR ---
# stock_list = get_all_stock_symbols()
# selected_ticker = st.sidebar.selectbox("Select Stock", stock_list)

# --- UTILITY FUNCTIONS ---
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
                l = df['Low'].iloc[i]
                if np.sum([abs(l - x) < (current_price*0.02) for x in levels]) == 0:
                    levels.append(l)
                    level_count += 1
            elif is_resistance(df, i, n):
                l = df['High'].iloc[i]
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

# --- COMPARISON (FIXED) ---
@st.cache_data(ttl=3600)
def get_market_comparison(ticker, period):
    try:
        tickers = [ticker, "^NSEI"]
        # Explicitly select 'Close' and handle multi-indexing
        data = yf.download(tickers, period=period, progress=False)['Close']
        
        if data.empty:
            return None, 0
            
        # Ensure it's a DataFrame and not a Series
        if isinstance(data, pd.Series):
            return None, 0

        normalized = (data / data.iloc[0] - 1) * 100
        correlation = data.corr().iloc[0, 1]
        return normalized, correlation
    except Exception as e:
        return None, 0


import yfinance as yf
from nsepython import nse_eq

@st.cache_data(ttl=86400)
@st.cache_data(ttl=86400)


# --- HELPER: FORMAT MARKET CAP ---
def format_market_cap(value):
    """Converts raw number to 'Lakh Cr' format."""
    if value == 0:
        return "N/A"
    
    # Value is usually in Rupees (absolute)
    crores = value / 10000000  # Convert to Crores
    
    if crores >= 100000:
        return f"₹{crores/100000:.2f} Lakh Cr"
    else:
        return f"₹{crores:.0f} Cr"


# --- 2. YOUR MAIN FUNCTION (With the Backup added at the end) ---


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

# --- DATA FETCHING (FIXED) ---
@st.cache_data(ttl=300) 
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    
    # Fetch data
    df = stock.history(period=period)
    
    # FIX: If yfinance returns multi-index columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty: 
        # Return an empty DF with columns and a default dict to avoid unpacking errors
        empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        return empty_df, {"sector": "N/A", "pe": "N/A", "mcap": "N/A"}
    
    # --- FUNDAMENTALS ---
    fundamentals = {"sector": "N/A", "pe": "N/A", "mcap": "N/A"}
    try:
        info = stock.info
        fundamentals["sector"] = info.get('sector', 'N/A')
        fundamentals["pe"] = info.get('trailingPE', 'N/A')
        fundamentals["mcap"] = info.get('marketCap', 'N/A')
    except:
        pass

    # --- TECHNICALS ---
    # Use .copy() to avoid SettingWithCopy warnings
    df = df.copy()
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
    
    return df, fundamentals

# --- SIDEBAR & CONTROLS ---
st.sidebar.markdown('<div class="sidebar-brand">⬡ &nbsp;Terminal</div>', unsafe_allow_html=True)

search_query = st.sidebar.text_input("Search by Name", placeholder="e.g. Reliance, Infosys...")

if search_query:
    # Yahoo search — matches company names
    results = search_tickers(search_query)
    if results:
        selected_label = st.sidebar.selectbox("Select", options=list(results.keys()))
        ticker = results[selected_label]
        st.session_state["ticker"] = ticker
    else:
        st.sidebar.caption("No matches. Clear to browse all stocks.")
        ticker = st.session_state.get("ticker", "RELIANCE.NS")
else:
    # Browse all NSE stocks — selectbox has built-in filter-as-you-type
    all_symbols = get_all_stock_symbols()
    current_symbol = st.session_state.get("ticker", "RELIANCE.NS").replace(".NS", "").replace(".BO", "")
    default_idx = all_symbols.index(current_symbol) if current_symbol in all_symbols else 0
    selected_symbol = st.sidebar.selectbox("Browse All Stocks", options=all_symbols, index=default_idx)
    ticker = f"{selected_symbol}.NS"
    st.session_state["ticker"] = ticker

period = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

if ticker:
    st.sidebar.markdown(f'<div class="section-label">Selected · {ticker}</div>', unsafe_allow_html=True) 
    
    # 1. Fetch info using the NEW NSE helper (Ensure you have renamed/updated your function)
    info = get_nse_fundamentals(ticker)
    
    # 2. Initialize safe defaults to prevent crashes
    sector = "N/A"
    pe_ratio = "N/A"
    market_cap = 0
    div_yield = 0.0

    if info:
        # UPDATED: Using NSE-specific keys
        sector = info.get('sector', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', 0)
        
        # Safely handle dividend yield percentage
        dy = info.get('dividendYield')
        if isinstance(dy, (int, float)):
            # NSE usually returns the actual percentage (e.g., 1.5)
            div_yield = dy 
        else:
            div_yield = 0.0
    
    # 3. SAFE P/E FORMATTING
    if isinstance(pe_ratio, (int, float)) and pe_ratio > 0:
        pe_str = f"{pe_ratio:.2f}"
    else:
        pe_str = "N/A"

    # 4. SAFE MARKET CAP FORMATTING (NSE returns values in Crores)
    # 4. SAFE MARKET CAP FORMATTING (Fix for NSE Scale)
    if isinstance(market_cap, (int, float)) and market_cap > 0:
        # If the value from NSE is already large, it's likely in Crores
        if market_cap >= 100000: 
            mcap_str = f"₹{market_cap/100000:.2f} Lakh Cr"
        else:
            mcap_str = f"₹{market_cap:,.2f} Cr"
    else:
        mcap_str = "N/A"

    # 5. UI DISPLAY
    st.sidebar.info(f"**Sector:** {sector}")
    formatted_mcap = format_market_cap(market_cap)
    st.sidebar.metric("Market Cap", formatted_mcap)
    st.sidebar.metric("P/E Ratio", pe_str)
    st.sidebar.metric("Div Yield", f"{div_yield:.2f}%")
else:
    st.sidebar.warning("Select a company to see fundamentals.")

st.sidebar.markdown('<div class="section-label">Portfolio</div>', unsafe_allow_html=True)

# 1. Form to Add Stock
with st.sidebar.expander("Add Stock to Portfolio"):
    # Auto-fill with the currently selected ticker
    p_ticker = st.text_input("Ticker", value=st.session_state.get("ticker", ""))
    p_qty = st.number_input("Quantity", min_value=1, value=10)
    p_price = st.number_input("Avg Buy Price", min_value=0.0, value=0.0)

    if st.sidebar.button("Add to Portfolio"):
        db.add_stock(p_ticker, p_qty, p_price)
        st.sidebar.success(f"Added {p_ticker}!")
        st.rerun() # Refresh app to show new data

# 2. Display Portfolio
portfolio_df = db.get_portfolio()

if not portfolio_df.empty:
    st.sidebar.write("### Your Holdings")

    # Calculate Current Value (Optional: Connects to your live data)
    # For now, just showing the DB table
    st.sidebar.dataframe(portfolio_df[['ticker', 'quantity']], hide_index=True)

    # Delete Button Logic
    stock_to_delete = st.sidebar.selectbox("Remove Stock", options=portfolio_df['ticker'])
    if st.sidebar.button("Delete Stock"):
        db.delete_stock(stock_to_delete)
        st.sidebar.warning(f"Removed {stock_to_delete}")
        st.rerun()
else:
    st.sidebar.info("Portfolio is empty.")
st.sidebar.markdown('<div class="section-label">Chart Display</div>', unsafe_allow_html=True)
chart_type = st.sidebar.selectbox("Chart Style", ["Candlestick", "Line", "Area", "OHLC"])

st.sidebar.markdown('<div class="section-label">AI Analyst</div>', unsafe_allow_html=True)
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
                st.markdown(f'<div class="section-label">AI Analyst · {ticker}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-card">{analysis}</div>', unsafe_allow_html=True)
                
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
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price Action", "📊 Technical Indicators", "⚖️ Sector Comparison", "💼 Portfolio", "👀 Watchlist"])
        
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
                         l = df['Low'].iloc[i]
                         if np.sum([abs(l - x[1]) < (current_price*0.02) for x in levels]) == 0:
                             levels.append((df.index[i], l, "Support"))
                     elif show_resistance and is_resistance(df, i, n):
                         l = df['High'].iloc[i]
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
            
            # --- FETCH DATA EARLY (Needed for both Manage & Display sections) ---
            pf_df = db.get_portfolio()
            
            # --- SECTION 1: ADD NEW POSITION (Exact Old Layout) ---
            with st.expander("➕ Add New Position", expanded=False):
                st.caption("Step 1: Search & Select Stock")
                
                row1_1, row1_2 = st.columns(2)
                with row1_1:
                    pf_search = st.text_input("Search Ticker", placeholder="e.g. TCS", key="pf_add_search")
                
                chosen_ticker = None
                with row1_2:
                    if pf_search:
                        results = search_tickers(pf_search) 
                        if results:
                            selected_label = st.selectbox("Select Stock", options=results.keys(), key="pf_add_select")
                            chosen_ticker = results[selected_label]
                        else:
                            st.warning("No stocks found.")

                # Auto-fill logic (Preserved)
                if chosen_ticker:
                    if 'last_pf_ticker' not in st.session_state: st.session_state.last_pf_ticker = None
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

                # Layout: Inputs + Button in one row (Preserved from your code)
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
                    if st.button("Add"):
                        if new_ticker and new_price > 0:
                            db.add_stock(new_ticker, new_qty, new_price)
                            st.toast(f"Added {new_ticker}!", icon="✅")
                            st.rerun()
                        else:
                            st.error("Invalid")
        with tab5:
            st.subheader("👀 My Watchlist")
            
            # --- SECTION 1: ADD TO WATCHLIST ---
            with st.expander("➕ Add to Watchlist", expanded=False):
                st.caption("Search & Select Stock to Monitor")
                
                row_w1, row_w2 = st.columns(2)
                with row_w1:
                    wl_search = st.text_input("Search Ticker", placeholder="e.g. INFY", key="wl_add_search")
                
                chosen_wl_ticker = None
                with row_w2:
                    if wl_search:
                        results = search_tickers(wl_search) 
                        if results:
                            selected_label = st.selectbox("Select Stock", options=results.keys(), key="wl_add_select")
                            chosen_wl_ticker = results[selected_label]
                        else:
                            st.warning("No stocks found.")
                
                if chosen_wl_ticker:
                    if st.button("Add to Watchlist", type="primary"):
                        db.add_to_watchlist(chosen_wl_ticker)
                        st.success(f"Added {chosen_wl_ticker} to Watchlist!")
                        st.rerun()

            # --- SECTION 2: DISPLAY LIVE WATCHLIST ---
            watchlist_tickers = db.get_watchlist()
            
            if not watchlist_tickers:
                st.info("Your watchlist is empty. Search and add some stocks above to start tracking them!")
            else:
                st.write("### Live Tracking")
                
                # Fetch quick live data for the grid
                watch_data = []
                with st.spinner("Fetching live prices..."):
                    for t in watchlist_tickers:
                        try:
                            # Fetch last 2 days to calculate % change
                            hist = yf.Ticker(t).history(period="2d")
                            if not hist.empty:
                                current_p = hist['Close'].iloc[-1]
                                prev_p = hist['Close'].iloc[-2] if len(hist) > 1 else current_p
                                pct_change = ((current_p - prev_p) / prev_p) * 100
                                
                                # Use formatting for clean display
                                watch_data.append({
                                    "Ticker": t, 
                                    "LTP (₹)": f"{current_p:.2f}", 
                                    "Change %": f"{pct_change:+.2f}%"
                                })
                        except Exception:
                            watch_data.append({"Ticker": t, "LTP (₹)": "Error", "Change %": "Error"})
                
                # Render the Data Table
                if watch_data:
                    wl_df = pd.DataFrame(watch_data)
                    # Pandas styling to color-code the % change (Green for positive, Red for negative)
                    def color_change(val):
                        if "Error" in str(val): return ''
                        color = '#00e676' if '+' in str(val) else '#ff1744' if '-' in str(val) else 'white'
                        return f'color: {color}'
                    
                    styled_df = wl_df.style.map(color_change, subset=['Change %'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # --- SECTION 3: REMOVE FROM WATCHLIST ---
                st.divider()
                st.caption("Manage List")
                col_rem1, col_rem2 = st.columns([3, 1])
                with col_rem1:
                    stock_to_remove = st.selectbox("Remove a stock", options=watchlist_tickers, key="wl_remove")
                with col_rem2:
                    st.write("") # Spacing to align with selectbox
                    st.write("")
                    if st.button("Delete"):
                        db.remove_from_watchlist(stock_to_remove)
                        st.warning(f"Removed {stock_to_remove} from Watchlist.")
                        st.rerun()
            # --- SECTION 2: MANAGE HOLDINGS (New Feature) ---
            with st.expander("✏️ Manage Holdings (Edit / Delete)", expanded=False):
                if not pf_df.empty:
                    # 1. Select Stock
                    stock_list = pf_df['ticker'].tolist()
                    selected_stock = st.selectbox("Select Stock to Edit", stock_list, key="manage_select")
                    
                    # 2. Get Current Values
                    current_row = pf_df[pf_df['ticker'] == selected_stock].iloc[0]
                    cur_qty = int(current_row['quantity'])
                    cur_price = float(current_row['avg_price'])
                    
                    # 3. Edit Form
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        edit_qty = st.number_input("New Quantity", min_value=1, value=cur_qty, key="edit_qty")
                    with m2:
                        edit_price = st.number_input("New Price", min_value=0.0, value=cur_price, format="%.2f", key="edit_price")
                    with m3:
                        st.write("")
                        st.write("")
                        if st.button("Update Position"):
                            db.update_stock(selected_stock, edit_qty, edit_price)
                            st.toast(f"Updated {selected_stock}!", icon="🔄")
                            st.rerun()

                    # 4. Single Delete Button
                    if st.button(f"🗑️ Remove {selected_stock} Only", type="secondary"):
                        db.delete_stock(selected_stock)
                        st.rerun()
                else:
                    st.info("Portfolio is empty.")

            # --- SECTION 3: DISPLAY TABLE ---
            if not pf_df.empty:
                # Rename for UI (We use a copy 'display_df' so we don't break the logic above)
                display_df = pf_df.rename(columns={"ticker": "Ticker", "quantity": "Quantity", "avg_price": "Buy Price"})
                
                realtime_prices = {}
                unique_tickers = display_df['Ticker'].unique()

                # Fetch Live Prices
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
                    except:
                        realtime_prices[t] = 0.0

                # Calculations
                display_df['Current Price'] = display_df['Ticker'].map(realtime_prices).fillna(0.0)
                display_df['Invested Value'] = display_df['Quantity'] * display_df['Buy Price']
                display_df['Current Value'] = display_df['Quantity'] * display_df['Current Price']
                display_df['P/L'] = display_df['Current Value'] - display_df['Invested Value']
                display_df['P/L %'] = (display_df['P/L'] / display_df['Invested Value']) * 100

                total_invested = display_df['Invested Value'].sum()
                total_current = display_df['Current Value'].sum()
                total_pl = display_df['P/L'].sum()
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Invested", f"₹{total_invested:,.2f}")
                m2.metric("Current Value", f"₹{total_current:,.2f}")
                m3.metric("Total P/L", f"₹{total_pl:,.2f}", delta=f"{total_pl:,.2f}")

                # Table
                st.dataframe(display_df.style.format({
                    "Buy Price": "₹{:.2f}", 
                    "Current Price": "₹{:.2f}",
                    "Invested Value": "₹{:.2f}",
                    "Current Value": "₹{:.2f}",
                    "P/L": "₹{:.2f}",
                    "P/L %": "{:.2f}%"
                }).map(lambda x: 'color: #00e676' if x > 0 else 'color: #ff1744', subset=['P/L', 'P/L %']),
                use_container_width=True)
                
                # --- SECTION 4: CLEAR ALL BUTTON (Restored) ---
                st.divider()
                if st.button("Clear Entire Portfolio", type="primary"):
                     for t in unique_tickers:
                         db.delete_stock(t)
                     st.rerun()
                     
            else:
                st.info("Portfolio is empty.")
else:
    st.error("Error loading data.")