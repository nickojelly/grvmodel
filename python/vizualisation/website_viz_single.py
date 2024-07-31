import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('single_day_for_viz.csv')  # Replace with your actual file path

# Prepare the race IDs
race_ids = df['raceId'].unique()

# Prepare the data columns
data_cols = ['place_sft', 'margin_sft', 'bosted_sft', 'bsp_prob']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Race Analysis Dashboard", className="title"),
        dcc.Dropdown(
            id='race-selector',
            options=[{'label': f'Race {i}', 'value': i} for i in race_ids],
            value=race_ids[0],
            className="dropdown"
        ),
    ], className="container"),

    html.Div([
        html.H2("Race Results", className="title"),
        dash_table.DataTable(
            id='table',
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': '#3498db',
                'color': 'white',
                'fontWeight': 'bold',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                }
            ]
        )
    ], className="container"),

    html.Div([
        html.H2("Result Margins", className="title"),
        dcc.Graph(id='column-chart')
    ], className="chart-container"),

    html.Div([
        html.H2("Race Heatmap", className="title"),
        dcc.Graph(id='heatmap')
    ], className="chart-container")
])

@app.callback(
    [Output('table', 'columns'), Output('table', 'data'), 
     Output('column-chart', 'figure'), Output('heatmap', 'figure')],
    [Input('race-selector', 'value')]
)
def update_content(race_index):
    # Get data for the selected race
    single_race = df[df['raceId'] == race_index]

    # Ensure we have data for all 8 dogs
    dogs = [f'Dog{i+1}' for i in range(8)]
    
    # Prepare data for heatmap
    data = np.zeros((len(data_cols), 8))  # Initialize with zeros for 8 dogs
    
    for i, row in single_race.iterrows():
        dog_index = int(row['boxNumber']) - 1  # Assuming boxNumber starts from 1
        data[:, dog_index] = row[data_cols]

    # Prepare data for bar chart
    result_time = single_race['resultMargin'].values
    
    # Update the column chart
    column_chart = {
        'data': [go.Bar(x=result_time, y=dogs, orientation='h')],
        'layout': go.Layout(
            title='Result Margins',
            xaxis_title='Margin',
            yaxis_title='Dog',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
        )
    }

    # Update the heatmap
    heatmap = {
        'data': [go.Heatmap(
            z=data,
            x=dogs,
            y=data_cols,
            text=data.round(3),
            hoverinfo='text',
            colorscale='YlGnBu',
        )],
        'layout': go.Layout(
            title='Race Heatmap',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
        )
    }

    return (
        [{"name": i, "id": i} for i in single_race[['boxNumber','win','resultMargin','place','resultTime','BSP']].columns],
        single_race[['boxNumber','win','resultMargin','place','resultTime','BSP']].round(3).to_dict('records'),
        column_chart,
        heatmap
    )

if __name__ == '__main__':
    app.write_html("index.html")