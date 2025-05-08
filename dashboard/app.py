# dashboard/app.py

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import os

external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Streamer Sentiment Monitor"
csv_path = "output/classified_chats.csv"

app.layout = html.Div(className="container-fluid bg-dark text-white p-4", children=[
    html.H2("🎮 Real-Time Stream Sentiment & Satisfaction", className="text-center mb-4"),

    html.Div(id="live-count", className="text-center h4 mb-4"),

    html.Div(className="row", children=[
        html.Div(className="col-md-6", children=[
            dcc.Graph(id="sentiment-graph")
        ]),
        html.Div(className="col-md-6", children=[
            dcc.Graph(id="satisfaction-graph")
        ])
    ]),

    html.H4("Recent Chat Logs", className="mt-4"),
    dash_table.DataTable(
        id="live-table",
        columns=[
            {"name": "Timestamp", "id": "timestamp"},
            {"name": "Platform", "id": "platform"},
            {"name": "Text", "id": "text"},
            {"name": "Sentiment", "id": "sentiment"},
            {"name": "Satisfaction", "id": "satisfaction"}
        ],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#343a40", "color": "white"},
        style_cell={"backgroundColor": "#212529", "color": "white"},
        page_size=5
    ),

    dcc.Interval(id="interval", interval=5000, n_intervals=0)
])

@app.callback(
    [Output("sentiment-graph", "figure"),
     Output("satisfaction-graph", "figure"),
     Output("live-table", "data"),
     Output("live-count", "children")],
    [Input("interval", "n_intervals")]
)
def update_dashboard(_):
    if not os.path.exists(csv_path):
        return dash.no_update

    df = pd.read_csv(csv_path)
    if df.empty:
        return dash.no_update

    sentiment_fig = px.pie(df, names="sentiment", title="Sentiment Pie", hole=0.3)
    satisfaction_fig = px.pie(df, names="satisfaction", title="Satisfaction Pie", hole=0.3)

    recent_data = df.tail(5)[["timestamp", "platform", "text", "sentiment", "satisfaction"]].iloc[::-1].to_dict("records")
    count_str = f"Total Posts Streamed: {len(df)}"

    return sentiment_fig, satisfaction_fig, recent_data, count_str

if __name__== "_main_":
    app.run_server(debug=True)