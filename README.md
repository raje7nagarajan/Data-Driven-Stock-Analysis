# Data-Driven-Stock-Analysis
This project aims to build a full pipeline for the extraction, processing, analysis, and performance visualization of Nifty 50 stocks for the previous year. For some user interactivity, Streamlit and Power BI dashboards have been developed along with other data processing scripts to assist investors, analysts, and enthusiasts in making informed decisions.'

**Data Extraction & Storage(StockConversion.ipynb)**

* Extract daily stock data from YAML files.
* Stores the processed data into a SQL database.
* Generates 50 CSV files, one for each stock symbol, also one CSV file to store all the data.

**Visual Analytics using Streamlit Dashboard(Stockviz.py)**

* Top 10 Green & Red Stocks: Based on yearly return.
* Volatility Analysis taken out with respect to the top 10 most volatile stocks by computing the standard deviation of daily returns.
* Cumulative Return is computed to visualize the top 5 performing stocks by printing them in a line chart.
* Sector-wise Performance is done based on the average yearly return by sector.
* Stock Correlation is displayed using the heatmap by showing the correlation matrix of closing prices.
* Monthly Gainers/Losers information is calculated for the month and represented in bar graphs for the top 5 gainers and losers.

**Visual Analytics using Power BI Dashboard(Data_Driven.pbix)**

* An alternative visualization tool showing the same metrics for professional reporting.
* Designed for business analysts and financial stakeholders.
* Separate pages were added to show the visualization for volatility analysis, cumulative return, sector-wise performance, stock correlation and monthly gainers/losers.

**File Details:**
* The .ipynb notebook handles all data extraction, transformation, and SQL loading.
* The .py Streamlit script builds interactive visualizations.
* Power BI file .pbix offers an interactive report view for analysis.
* The processed CSV files were added for reference purposes.

**Tools & Technologies**
* **Language:** Python
* **Libraries:** Pandas, Matplotlib, MySQL connector, Streamlit, YAML
* **Database:** MySQL(XAMPP)
* **Visualization Tools:** Streamlit, Power BI
  
