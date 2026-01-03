import sqlite3
import pandas as pd

# Name of our database file
DB_NAME = "portfolio.db"

def init_db():
    """Creates the table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            quantity INTEGER NOT NULL,
            avg_price REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_stock(ticker, quantity, avg_price):
    """Adds a new stock or updates existing one."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Check if stock exists
    c.execute('SELECT quantity, avg_price FROM portfolio WHERE ticker = ?', (ticker,))
    data = c.fetchone()
    
    if data:
        # Stock exists: Update Weighted Average Price & Total Qty
        old_qty, old_price = data
        new_total_qty = old_qty + quantity
        new_avg_price = ((old_qty * old_price) + (quantity * avg_price)) / new_total_qty
        
        c.execute('UPDATE portfolio SET quantity = ?, avg_price = ? WHERE ticker = ?', 
                  (new_total_qty, new_avg_price, ticker))
    else:
        # New Stock: Insert it
        c.execute('INSERT INTO portfolio (ticker, quantity, avg_price) VALUES (?, ?, ?)', 
                  (ticker, quantity, avg_price))
        
    conn.commit()
    conn.close()

def get_portfolio():
    """Returns the portfolio as a Pandas DataFrame."""
    conn = sqlite3.connect(DB_NAME)
    # Pandas makes it easy to read SQL directly into a table
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
    conn = sqlite3.connect('portfolio.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        UPDATE portfolio 
        SET quantity = ?, avg_price = ? 
        WHERE ticker = ?
    ''', (new_quantity, new_avg_price, ticker))
    conn.commit()
    conn.close()