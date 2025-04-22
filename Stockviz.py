# --- REQUIRED HEADER FILES ---
import streamlit as st
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. DATABASE CONNECTION AND DATA PROCESSING ---
def load_data():
    """
    Connects to the MySQL database and loads stock and sector data into pandas DataFrames.
    """
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database="StockData"
    )

    # Load stock price data
    stock_query = "SELECT ticker, date, close FROM stock"
    df = pd.read_sql(stock_query, conn)

    # Load sector data
    sector_query = "SELECT symbol, sector FROM sector"
    sectors_df = pd.read_sql(sector_query, conn)

    conn.close()
    return df, sectors_df


# --- 2. DATA PREPROCESSING ---
def preprocess_data(df):
    """
    Preprocesses the stock data:
    - Converts date to datetime format
    - Sorts by ticker and date
    - Calculates monthly period and daily returns
    """
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df = df.sort_values(by=['ticker', 'date'])
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    return df


# --- 3. VOLATILITY ANALYSIS ---
def show_volatility(df):
    """
    Displays the top 10 most volatile stocks based on standard deviation of daily returns.
    """
    st.markdown("<h4 style='color:black;'>1. Top 10 Most Volatile Stocks</h4>", unsafe_allow_html=True)

    # Calculate standard deviation of daily returns
    volatility = df.groupby('ticker')['daily_return'].std().reset_index()
    volatility.columns = ['ticker', 'volatility']

    # Sort and select top 10
    top_10 = volatility.sort_values(by='volatility', ascending=False).head(10)
    top_10.to_csv("volatility_data.csv", index=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_10['ticker'], top_10['volatility'], color='blue')
    ax.set_title("Top 10 Most Volatile Stocks")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Volatility (Standard Deviation)")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)


# --- 4. CUMULATIVE RETURN ANALYSIS ---
def show_cumulative_return(df):
    """
    Displays cumulative returns over time for the top 5 performing stocks.
    """
    st.markdown("<h4 style='color:black;'>2. Cumulative Return Over Time</h4>", unsafe_allow_html=True)

    # Calculate cumulative return per stock
    df['cumulative_return'] = (1 + df['daily_return']).groupby(df['ticker']).cumprod() - 1

    # Get latest cumulative return per ticker
    end_returns = df.groupby('ticker').last()['cumulative_return']
    top_5 = end_returns.sort_values(ascending=False).head(5).index

    # Filter top 5 stocks
    top_df = df[df['ticker'].isin(top_5)]
    top_df.to_csv("cumulative_return.csv", index=False)

    # Line plot of cumulative returns
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=top_df, x='date', y='cumulative_return', hue='ticker', palette='tab10', ax=ax)
    ax.set_title("Cumulative Return for Top 5 Performing Stocks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    st.pyplot(fig)


# --- 5. SECTOR PERFORMANCE ANALYSIS ---
def show_sector_performance(df, sectors_df):

    """
    Analyze sector-wise performance by computing the average yearly return per sector,
    plotting the results, and saving them to a CSV.
    """

    st.markdown("<h4 style='color:black;'>3. Sector-wise Performance</h4>", unsafe_allow_html=True)

    # Sort the data to calculate daily returns correctly
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['ticker', 'date'])

    # Calculate daily returns: (Close - Previous Close) / Previous Close
    df['Daily_Return'] = df.groupby('ticker')['close'].pct_change()
    print("\nStock Data with Daily Returns:")
    print(df[['ticker', 'date', 'close', 'Daily_Return']].head(10))

    # Calculate average yearly return for each stock
    yearly_return_df = df.groupby(['ticker'])['Daily_Return'].mean().reset_index()
    yearly_return_df.rename(columns={'Daily_Return': 'Avg_Yearly_Return'}, inplace=True)
    print("\nAverage Yearly Returns per Stock:")
    print(yearly_return_df.head())

    # Clean up Sector data before merging
    sectors_df['ticker'] = sectors_df['symbol'].str.split(': ').str[-1].str.strip()
    sectors_df = sectors_df[['ticker', 'sector']]
    sectors_df.rename(columns={'sector': 'Sector'}, inplace=True)
    print("\nCleaned Sector Data:")
    print(sectors_df.head())

    # Merge average returns with cleaned sector data
    merged_df = pd.merge(yearly_return_df, sectors_df, on='ticker', how='left')
    print("\nMerged Data:")
    print(merged_df.head())

    # Group by Sector to calculate average yearly return per sector
    sector_avg_returns = merged_df.groupby('Sector')['Avg_Yearly_Return'].mean().reset_index()

    # Optional: Convert to percentage for easier interpretation
    sector_avg_returns['Avg_Yearly_Return'] *= 100
    print("\nSector-wise Average Yearly Returns:")
    print(sector_avg_returns)

    # Save the result to a CSV file
    output_csv_path = 'sector_avg_returns.csv'
    sector_avg_returns.to_csv(output_csv_path, index=False)
    print(f"\nSector-wise average returns saved to: {output_csv_path}")

    # Plotting
    sector_avg_returns = sector_avg_returns.sort_values(by='Avg_Yearly_Return', ascending=False)
    fig_sector, ax_sector = plt.subplots(figsize=(12, 6))
    ax_sector.bar(sector_avg_returns['Sector'], sector_avg_returns['Avg_Yearly_Return'], color='steelblue')
    ax_sector.set_xlabel('Sector')
    ax_sector.set_ylabel('Average Yearly Return (%)')
    ax_sector.set_title('Sector-wise Average Yearly Return')
    ax_sector.tick_params(axis='x', rotation=50)
    ax_sector.grid(axis='y', linestyle='--', alpha=0.6)
    fig_sector.tight_layout()
    st.pyplot(fig_sector)

    return sector_avg_returns

# --- 6. CORRELATION HEATMAP ---
def show_correlation(df):
    """
    Displays a heatmap of correlation between stock prices.
    """
    st.markdown("<h4 style='color:black;'>4. Stock Price Correlation</h4>", unsafe_allow_html=True)

    # Pivot data for correlation matrix
    pivot_df = df.pivot(index='date', columns='ticker', values='close')
    if pivot_df.shape[1] < 2:
        st.warning("Need at least two stocks to show correlation.")
        return

    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    corr_matrix.to_csv("correlation_matrix.csv")

    # Melt for long format CSV
    melted_corr = pd.melt(corr_matrix.reset_index(), id_vars="ticker", var_name="cticker", value_name="Correlation")
    melted_corr.to_csv("correlationmat.csv", index=False)

    # Plot heatmap
    plt.figure(figsize=(25, 20))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, annot_kws={"size": 6})
    plt.title("Stock Price Correlation Heatmap", fontsize=16)
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)


# --- 7. TOP GAINERS & LOSERS PER MONTH ---
def show_monthly_gainers_losers(df):
    """
    For each month, displays top 5 gainers and losers based on percentage return.
    """
    st.markdown("<h4 style='color:black;'>5. Top 5 Gainers and Losers (Month-wise)</h4>", unsafe_allow_html=True)

    # Calculate monthly returns
    monthly = df.groupby(['ticker', 'month'])['close'].agg(['first', 'last']).reset_index()
    monthly['pct_return'] = ((monthly['last'] - monthly['first']) / monthly['first']) * 100
    months = monthly['month'].unique()

    for month in months:
        month_data = monthly[monthly['month'] == month]
        top_5 = month_data.nlargest(5, 'pct_return')
        bottom_5 = month_data.nsmallest(5, 'pct_return')

        # Section title per month
        st.subheader(f"{month}")
        col1, col2 = st.columns(2)

        # Append to CSV
        top_5.to_csv("top_5_gainers.csv", mode='a', header=not os.path.exists("top_5_gainers.csv"), index=False)
        bottom_5.to_csv("top_5_losers.csv", mode='a', header=not os.path.exists("top_5_losers.csv"), index=False)

        # Top Gainers Plot
        with col1:
            st.write("Top 5 Gainers")
            fig, ax = plt.subplots()
            sns.barplot(data=top_5, x='ticker', y='pct_return', palette="Greens_d", ax=ax)
            ax.set_title("Top 5 Gainers")
            st.pyplot(fig)

        # Top Losers Plot
        with col2:
            st.write("Top 5 Losers")
            fig, ax = plt.subplots()
            sns.barplot(data=bottom_5, x='ticker', y='pct_return', palette="Reds_d", ax=ax)
            ax.set_title("Top 5 Losers")
            st.pyplot(fig)


# --- 8. MAIN STREAMLIT APP ---
def main():
    """
    Display the full Streamlit dashboard by calling all analysis components.
    """
    st.markdown("<h1 style='text-align: center; color:orange;'>Data-Driven Stock Analysis</h1>", unsafe_allow_html=True)

    # Load and preprocess data
    df, sectors_df = load_data()
    df = preprocess_data(df)

    # Show each section of the dashboard
    show_volatility(df)
    show_cumulative_return(df)
    show_sector_performance(df, sectors_df)
    show_correlation(df)
    show_monthly_gainers_losers(df)


if __name__ == "__main__":
    main()