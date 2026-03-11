import sqlite3
import pandas as pd

# Name of our database file
DB_NAME = "portfolio.db"

def init_db():
    """Creates the tables if they don't exist yet."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. Existing Portfolio Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            quantity INTEGER NOT NULL,
            avg_price REAL NOT NULL
        )
    ''')
    
    # 2. NEW: Watchlist Table
    # (Note: We only need the ticker. We'll fetch live data for these in the UI)
    c.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE
        )
    ''')
    
    conn.commit()
    conn.close()

# --- PORTFOLIO FUNCTIONS ---

def add_stock(ticker, quantity, avg_price):
    """Adds a new stock or updates existing one."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT quantity, avg_price FROM portfolio WHERE ticker = ?', (ticker,))
    data = c.fetchone()
    
    if data:
        old_qty, old_price = data
        new_total_qty = old_qty + quantity
        new_avg_price = ((old_qty * old_price) + (quantity * avg_price)) / new_total_qty
        c.execute('UPDATE portfolio SET quantity = ?, avg_price = ? WHERE ticker = ?', 
                  (new_total_qty, new_avg_price, ticker))
    else:
        c.execute('INSERT INTO portfolio (ticker, quantity, avg_price) VALUES (?, ?, ?)', 
                  (ticker, quantity, avg_price))
    conn.commit()
    conn.close()

def get_portfolio():
    """Returns the portfolio as a Pandas DataFrame."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM portfolio", conn)
    conn.close()
    return df

def delete_stock(ticker):
    """Removes a stock from the DB."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM portfolio WHERE ticker = ?', (ticker,))
    conn.commit()
    conn.close()

def update_stock(ticker, new_quantity, new_avg_price):
    """Updates the quantity and price of an existing stock."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        UPDATE portfolio 
        SET quantity = ?, avg_price = ? 
        WHERE ticker = ?
    ''', (new_quantity, new_avg_price, ticker))
    conn.commit()
    conn.close()

# --- NEW: WATCHLIST FUNCTIONS ---

def add_to_watchlist(ticker):
    """Adds a ticker to the watchlist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Use INSERT OR IGNORE to prevent errors if user adds the same ticker twice
        c.execute('INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)', (ticker,))
        conn.commit()
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
    finally:
        conn.close()

def get_watchlist():
    """Returns the watchlist as a list of tickers."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT ticker FROM watchlist')
    # Fetchall returns a list of tuples like [('AAPL',), ('TSLA',)], 
    # so we flatten it into a simple list ['AAPL', 'TSLA']
    tickers = [row[0] for row in c.fetchall()]
    conn.close()
    return tickers

def remove_from_watchlist(ticker):
    """Removes a ticker from the watchlist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM watchlist WHERE ticker = ?', (ticker,))
    conn.commit()
    conn.close()