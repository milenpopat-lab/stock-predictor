import streamlit as st
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Forecast App")

stocks = ("GOOG", "AAPL", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START,
        end=TODAY,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    df = df.reset_index()

    # If yfinance returns MultiIndex columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x is not None and str(x) != ""])
            for col in df.columns.to_list()
        ]

    # Drop duplicate column names if any
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def pick_1d_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Always return a 1-D Series for the requested column name.
    Handles duplicate col names / multiindex flattening edge cases.
    """
    if col_name in df.columns:
        obj = df[col_name]
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj

    # Try common flattened patterns (Close_AAPL, AAPL_Close, etc.)
    n = col_name.lower()
    candidates = [
        c for c in df.columns
        if str(c).lower() == n
        or str(c).lower().endswith(f"_{n}")
        or str(c).lower().startswith(f"{n}_")
        or n in str(c).lower()
    ]
    if not candidates:
        raise KeyError(f"Could not find '{col_name}' in columns: {list(df.columns)}")

    obj = df[candidates[0]]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

st.subheader("Raw data")
st.write(data.tail())

with st.expander("Debug: columns from yfinance"):
    st.write(list(data.columns))


# Plot raw data
def plot_raw_data():
    # yfinance usually gives "Date" after reset_index, but sometimes it can be "Datetime"
    date_col = "Date" if "Date" in data.columns else ("Datetime" if "Datetime" in data.columns else data.columns[0])

    open_s = pick_1d_series(data, "Open")
    close_s = pick_1d_series(data, "Close")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[date_col], y=open_s, name="stock_open"))
    fig.add_trace(go.Scatter(x=data[date_col], y=close_s, name="stock_close"))
    fig.update_layout(
        title_text="Time Series data with Rangeslider",
        xaxis_rangeslider_visible=True,
    )
    st.plotly_chart(fig, use_container_width=True)


plot_raw_data()

# -------------------------
# Forecast with Prophet (HARDENED: guarantees ds/y are 1-D and no duplicate 'y' column)
# -------------------------
date_col = "Date" if "Date" in data.columns else ("Datetime" if "Datetime" in data.columns else data.columns[0])

ds_series = pick_1d_series(data, date_col)
close_series = pick_1d_series(data, "Close")

# Convert to 1-D numpy arrays explicitly (this avoids pandas returning DataFrames on duplicate names)
ds_arr = pd.to_datetime(np.asarray(ds_series).reshape(-1), errors="coerce")
y_arr = pd.to_numeric(np.asarray(close_series).reshape(-1), errors="coerce")

# Build df_train from arrays so it can only have ONE 'y' column
df_train = pd.DataFrame({"ds": ds_arr, "y": y_arr}).dropna()

# Force columns to be exactly ['ds','y'] (no duplicates, no extras)
df_train = df_train.loc[:, ["ds", "y"]].copy()
df_train.columns = ["ds", "y"]

# Remove duplicate timestamps (Prophet prefers unique ds)
df_train = df_train.groupby("ds", as_index=False)["y"].mean()

with st.expander("Debug: Prophet training df (should be 2 cols: ds,y)"):
    st.write("columns:", list(df_train.columns))
    st.write("type(df_train['y']):", type(df_train["y"]))
    st.write(df_train.dtypes)
    st.write(df_train.head())

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())

st.write(f"Forecast plot for {n_years} years")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
