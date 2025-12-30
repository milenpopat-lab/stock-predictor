import streamlit as st
from datetime import date

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
    """Download historical prices from yfinance and normalize the columns."""
    df = yf.download(
        ticker,
        start=START,
        end=TODAY,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # yfinance sometimes uses "index" instead of "Date" after reset_index
    if "index" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"index": "Date"})

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x is not None and str(x) != ""])
            for col in df.columns.to_list()
        ]

    # Drop duplicate column names if any
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _candidate_cols(df: pd.DataFrame, col_name: str) -> list[str]:
    """Find columns that likely correspond to col_name (handles flattened/multiindex names)."""
    n = col_name.lower()
    cols: list[str] = []
    for c in df.columns:
        s = str(c).lower()
        if s == n or s.endswith(f"_{n}") or s.startswith(f"{n}_") or (n in s):
            cols.append(c)
    return cols


def pick_1d_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a 1-D Series for the requested column name."""
    if col_name in df.columns:
        obj = df[col_name]
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj

    candidates = _candidate_cols(df, col_name)
    if not candidates:
        raise KeyError(f"Could not find '{col_name}' in columns: {list(df.columns)}")

    obj = df[candidates[0]]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def pick_numeric_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a numeric (coerced) 1-D Series for the requested column name."""
    candidates: list[str] = []
    if col_name in df.columns:
        candidates.append(col_name)
    candidates.extend([c for c in _candidate_cols(df, col_name) if c not in candidates])

    if not candidates:
        raise KeyError(f"Could not find '{col_name}' in columns: {list(df.columns)}")

    best: pd.Series | None = None
    best_non_nan = -1

    for c in candidates:
        obj = df[c]
        s = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
        s_num = pd.to_numeric(s, errors="coerce")
        non_nan = int(s_num.notna().sum())

        if non_nan > best_non_nan:
            best_non_nan = non_nan
            best = s_num

        # Stop early if we found a clearly usable series
        if non_nan >= 10:
            return s_num

    # Fall back to the best we saw
    return best if best is not None else pd.Series(dtype="float64")


data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

if data.empty:
    st.error(
        "No data was returned from yfinance. This can happen due to rate limits or network issues.\n\n"
        "Try again in a minute, or try a different ticker."
    )
    st.stop()

st.subheader("Raw data")
st.write(data.tail())

with st.expander("Debug: columns from yfinance"):
    st.write(list(data.columns))


def plot_raw_data():
    date_col = "Date" if "Date" in data.columns else (
        "Datetime" if "Datetime" in data.columns else data.columns[0]
    )

    x = pd.to_datetime(data[date_col], errors="coerce")
    open_s = pick_numeric_series(data, "Open")
    close_s = pick_numeric_series(data, "Close")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=open_s, name="stock_open"))
    fig.add_trace(go.Scatter(x=x, y=close_s, name="stock_close"))
    fig.update_layout(
        title_text="Time Series data with Rangeslider",
        xaxis_rangeslider_visible=True,
    )
    st.plotly_chart(fig, use_container_width=True)


plot_raw_data()

# -------------------------
# Forecast with Prophet (robust + guarded)
# -------------------------
date_col = "Date" if "Date" in data.columns else (
    "Datetime" if "Datetime" in data.columns else data.columns[0]
)

ds = pd.to_datetime(data[date_col], errors="coerce")
y = pick_numeric_series(data, "Close")

df_train = pd.DataFrame({"ds": ds, "y": y}).dropna()
df_train = df_train.loc[:, ["ds", "y"]].copy()

# Remove duplicate timestamps (Prophet doesn't like duplicates)
df_train = df_train.groupby("ds", as_index=False)["y"].mean()

with st.expander("Debug: Prophet training df"):
    st.write("rows:", len(df_train))
    st.write(df_train.dtypes)
    st.write(df_train.head())
    st.write(df_train.tail())

if len(df_train) < 2:
    st.error(
        "Not enough non-NaN rows to train Prophet (need at least 2).\n\n"
        "This usually happens when the date column can't be parsed, or Close prices couldn't be converted to numbers."
    )
    st.stop()

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
