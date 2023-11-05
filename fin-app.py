import pycob as cob
import pandas as pd
import plotly.express as px
import yfinance as yf

app = cob.App("Portfolio Volatility")

initial_tickers_and_weights = {
    "AAPL": 0.2,
    "MSFT": 0.2,
    "AMZN": 0.2,
    "GOOG": 0.2,
    "META": 0.2,
}

def get_data(app: cob.App, tickers: list) -> pd.DataFrame:
    """Get the data for the tickers from Yahoo Finance"""
    data = pd.DataFrame()
    start_date = "2012-01-01"
    end_date = "2022-12-31"
    for ticker in tickers:
        ticker_data = yf.download(ticker, start=start_date, end=end_date)
        ticker_data['Ticker'] = ticker  # Add a column to identify the ticker
        data = pd.concat([data, ticker_data])

    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'date', 'Adj Close': 'closeadj', 'Ticker': 'ticker'}, inplace=True)
    return data
    

def update_tickers_and_weights(before: dict, ticker: str, weight: float) -> dict:
    after = before.copy()
    after[ticker] = weight

    # Normalize the weights
    total_weight = sum(after.values())

    for ticker in after:
        after[ticker] = after[ticker] / total_weight

    return after

def compute_daily_returns(data: pd.DataFrame) -> pd.DataFrame:
    data["return"] = data.groupby("ticker")["closeadj"].pct_change()

    return data

def compute_portfolio_returns(data: pd.DataFrame, tickers_and_weights: dict) -> pd.DataFrame:
    data['weighted_return'] = data['return'] * data['ticker'].map(tickers_and_weights)

    portfolio_return = data.groupby("date")["weighted_return"].sum()

    return portfolio_return

def get_return_vs_volatility(data: pd.DataFrame) -> pd.DataFrame:
    # First Date
    first_date = data["date"].min()

    # Last Date
    last_date = data["date"].max()

    # Ticker volatility
    ticker_volatility = data.groupby("ticker")["return"].std() * 252 ** 0.5

    # Compute the total returns using the first date and last date and closeadj
    total_returns = data[data["date"] == last_date].set_index("ticker")["closeadj"] / data[data["date"] == first_date].set_index("ticker")["closeadj"] - 1

    # Join ticker_volatility and total_returns
    return_vs_volatility = total_returns.to_frame("total_return").join(ticker_volatility.to_frame("volatility"))

    return return_vs_volatility

# Page Functions
def home(server_request: cob.Request) -> cob.Page:
    page = cob.Page("Portfolio Volatility")

    ticker = server_request.params("ticker")
    weight = server_request.params("weight")

    if ticker != "" and weight != "":
        # Update the tickers and weights
        tickers_and_weights = update_tickers_and_weights(initial_tickers_and_weights, ticker, float(weight))
    else:
        tickers_and_weights = initial_tickers_and_weights

    # Get the data
    data = get_data(server_request.app, tickers_and_weights.keys())
    symbols = data['ticker'].unique()
    print(symbols)
    data = compute_daily_returns(data)
    portfolio_return = compute_portfolio_returns(data, tickers_and_weights)

    # Calculate the portfolio volatility
    portfolio_volatility = portfolio_return.std() * 252 ** 0.5

    # Compute the portfolio cumulative return
    portfolio_cumulative_return = (1+portfolio_return).cumprod()-1

    # Compute portfolio Value at Risk
    tail_cutoff = data.groupby("date")["weighted_return"].sum().quantile(0.025)
    portfolio_var = tail_cutoff * 252 ** 0.5

    # Compute portfolio Expected Shortfall
    portfolio_es = portfolio_return[portfolio_return < tail_cutoff].mean() * 252 ** 0.5

    #calcuate sharpe ratio
    risk_free_rate = 0.0435
   # Calculate Sharpe Ratio (annualized for daily data)
    sharpe_ratio = (portfolio_return.mean() - risk_free_rate) / (portfolio_return.std() * (252 ** 0.5))

    print(sharpe_ratio)
    # Plotly Histogram of the returns
    fig1 = px.histogram(portfolio_return, x="weighted_return", nbins=100, title="<b>Daily Portfolio Return Distribution<b>")
    fig1.layout.xaxis.tickformat = ',.0%'
    
    # Plotly Cumulative Return
    fig2 = px.line(pd.DataFrame(portfolio_cumulative_return), x=portfolio_cumulative_return.index, y="weighted_return", title="<b>Portfolio Cumulative Return</b>")
    fig2.layout.yaxis.tickformat = ',.0%'

    # Plotly Scatter of the returns vs volatility
    return_vs_volatility = get_return_vs_volatility(data)
    fig3 = px.scatter(return_vs_volatility.reset_index(), x="volatility", y="total_return", title="<b>Returns vs Volatility</b>", text="ticker")
    fig3.update_traces(textposition="bottom right")

    # Compute the correlation matrix
    correlation_matrix = data.pivot(index="date", columns="ticker", values="return").corr()

    # Plotly Heatmap of the correlation matrix
    fig4 = px.imshow(correlation_matrix, title="<b>Correlation Matrix</b>")

    with page.add_container(grid_columns=4) as risk_stats:
        with risk_stats.add_card() as card:
            card.add_header(f"{portfolio_volatility:.2%}")
            card.add_text("Portfolio Volatility")
        
        with risk_stats.add_card() as card:
            card.add_header(f"{portfolio_var:.2%}")
            card.add_text("Portfolio Value at Risk")

        with risk_stats.add_card() as card:
            card.add_header(f"{portfolio_es:.2%}")
            card.add_text("Portfolio Expected Shortfall")

        with risk_stats.add_card() as card:
            card.add_header(f"{sharpe_ratio:.2%}")
            card.add_text("Sharpe Ratio")

    with page.add_container(grid_columns=2) as plots1:
        plots1.add_plotlyfigure(fig1)
        plots1.add_plotlyfigure(fig2)
    
    with page.add_container(grid_columns=2) as plots2:
        plots2.add_plotlyfigure(fig3)
        plots2.add_plotlyfigure(fig4)

    with page.add_card() as card:
        card.add_header("Portfolio Tickers and Weights")
        card.add_pandastable(pd.Series(tickers_and_weights).to_frame("weights").reset_index(names="tickers"))
        card.add_header("Add Ticker", size=3)
        card.add_text("The rest of the portfolio will be rebalanced to maintain the relative weights.")
        with card.add_form() as form:
            # Assuming `available_stocks` is a list of available stock tickers
            # Get the list of S&P 500 component stocks

            # Convert the index symbols to a list of strings

# Add other stocks or index funds to the list as
            available_stocks = symbols
            # Add a dropdown for selecting the ticker
            form.add_formselect("Ticker", "ticker", (available_stocks))

            # Add a text field for entering the weight
            form.add_formtext("Weight", "weight", "0.2")

            # Add a submit button to add the selected stock
            form.add_formsubmit("Add Ticker")


    return page



# tickers = ["AAPL", "MSFT", "GOOGL"]  # Replace with your list of tickers
# start_date = "2022-01-01"
# end_date = "2022-12-31"
# ticker_data = yf.download("AAPL", start=start_date, end=end_date)
# print(ticker_data)
# # Add the pages
app.register_function(home)

# Run the app
server = app.run() 