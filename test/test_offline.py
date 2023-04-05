import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import sys
sys.path.append("/home/nick/Time-Series-Regression/src")
from offline import batching


def line(df, x, y, color=None, title=None, font_size=None):
    fig = px.line(df, x=x, y=y, color=color, title=title)
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def histogram(df, x, bins=None, color=None, title=None, font_size=None):
    fig = px.histogram(df, x=x, nbins=bins, color=color, title=title, marginal="box")
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def series(df, predict, actual, color=None, title=None, font_size=None):
    df = df.reset_index()
    fig = px.line(df, x="index", y=predict, color=color, title=title)
    fig.add_trace(go.Scatter(x=df["index"], y=df[actual], mode="lines", showlegend=False, name="Actual"))
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def parity(df, predict, actual, color=None, title=None, font_size=None):
    fig = px.scatter(df, x=actual, y=predict, color=color, title=title)
    fig.add_trace(go.Scatter(x=df[actual], y=df[actual], mode="lines", showlegend=False, name="Actual"))
    fig.update_layout(font=dict(size=font_size))
    plot(fig)


data = pd.read_csv("/home/nick/Time-Series-Regression/test/traffic.txt", sep="\t")

# line(data, x="Day", y="Vehicles", font_size=16)
# histogram(data, x="Vehicles", bins=9, font_size=16)

model = batching(
    df=data, 
    datetime="Day", 
    output="Vehicles", 
    lags=7, 
    forecasts=7, 
    resolution=["day_of_week", "week_of_year"],
    test_frac=0.5,
)

forecast = 7
series(
    df=model.predictions[model.output[forecast - 1]],
    predict="Predicted",
    actual="Actual",
    font_size=16,
)

parity(
    df=model.predictions[model.output[forecast - 1]],
    predict="Predicted",
    actual="Actual",
    font_size=16,
)

print(model.metric[model.output[forecast - 1]])
