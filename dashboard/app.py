# dashboard/app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import base64
from io import BytesIO
import csv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.eda_utils import LiveEDA
from wordcloud import WordCloud

CSV_PATH = 'results/predictions_live.csv'

app = dash.Dash(__name__)
app.title = "Real-Time Sentiment Dashboard"

app.layout = html.Div([
    html.H1("📊 Real-Time Sentiment Analysis Dashboard", style={'textAlign': 'center'}),

    # Total Posts Analyzed
    html.Div([
        html.H4("Total Posts Analyzed:", style={'textAlign': 'center', 'marginTop': '20px'}),
        html.H2(id='total-count', style={'textAlign': 'center', 'color': '#2E86AB'})
    ]),

    # Bar Chart
    html.Div([
        dcc.Graph(id='sentiment-bar')
    ], style={'paddingTop': '30px'}),

    # WordCloud
    html.Div([
        html.H3("Live WordCloud", style={'textAlign': 'center', 'marginTop': '30px'}),
        html.Img(id='wordcloud-img', style={'display': 'block', 'margin': 'auto', 'width': '80%'})
    ]),

    # Most Recent 5 Posts
    html.Div([
        html.H4("Most Recent 5 Posts", style={'textAlign': 'center', 'marginTop': '30px'}),
        html.Div(id='last-tweets', style={
            'margin': 'auto',
            'width': '80%',
            'padding': '10px',
            'backgroundColor': '#F7F9FA',
            'borderRadius': '5px',
            'whiteSpace': 'pre-wrap',
            'fontFamily': 'monospace'
        })
    ], style={'paddingBottom': '30px'}),

    # Interval for updates
    dcc.Interval(
        id='interval-component',
        interval=10 * 1000,  # 10 seconds
        n_intervals=0
    )
])

def generate_wordcloud_base64(csv_path):
    eda = LiveEDA(csv_path)
    if not eda.texts:
        return None

    combined_text = ' '.join(eda.texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format='PNG')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{encoded_image}"

@app.callback(
    Output('total-count', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_total_count(n):
    try:
        df = pd.read_csv(CSV_PATH, quoting=csv.QUOTE_ALL, on_bad_lines='skip', encoding='utf-8')
        return f"{len(df)}"
    except:
        return "0"

@app.callback(
    Output('sentiment-bar', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_sentiment_bar(n):
    try:
        df = pd.read_csv(CSV_PATH, quoting=csv.QUOTE_ALL, on_bad_lines='skip', encoding='utf-8')
        df.columns = [c.strip().lower() for c in df.columns]
        if 'label' not in df.columns:
            raise ValueError("Missing 'label' column")

        counts = df['label'].value_counts().reset_index()
        counts.columns = ['Sentiment', 'Count']
        fig = px.bar(
            counts, x='Sentiment', y='Count',
            color='Sentiment', title='Live Sentiment Counts',
            text='Count'
        )
        fig.update_layout(xaxis_title=None, yaxis_title=None, plot_bgcolor='#F7F9FA')
        return fig
    except Exception as e:
        print("[Bar Chart Error]:", e)
        return px.bar(title="Waiting for data...")

@app.callback(
    Output('wordcloud-img', 'src'),
    [Input('interval-component', 'n_intervals')]
)
def update_wordcloud(n):
    return generate_wordcloud_base64(CSV_PATH)

@app.callback(
    Output('last-tweets', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_last_tweets(n):
    try:
        df = pd.read_csv(CSV_PATH, quoting=csv.QUOTE_ALL, on_bad_lines='skip', encoding='utf-8')
        df.columns = [c.strip().lower() for c in df.columns]
        if not all(col in df.columns for col in ['text', 'platform', 'label']):
            return "No data yet."

        last5 = df.tail(5).iloc[::-1]
        lines = []
        for _, row in last5.iterrows():
            lines.append(f"[{row['platform'].capitalize()}] ({row['label'].capitalize()}): {row['text']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"[Error loading tweets]: {e}"

if __name__ == '__main__':
    app.run(debug=True)
