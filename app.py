import dash
from dash import dcc, html, Input, Output
from flask_caching import Cache
import requests
import os

from ct_scan_dashboard import ct_scan_layout, register_callbacks
from ML import ML_layout, register_callback_ML

# Create the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server 

# Initialize Flask-Caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',  # Use 'redis' for Redis server in production
    'CACHE_DIR': 'cache-directory',  # For filesystem cache
    'CACHE_DEFAULT_TIMEOUT': 600  # Cache timeout (in seconds)
})

# Environment variable for LLM API URL
LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8000/generate")

# Main layout with dcc.Store initialized only once
app.layout = html.Div([
    dcc.Store(id='ct-scan-data-store', storage_type='memory'),  # Global store, available across all tabs
    dcc.Tabs(id='tabs', value='ct-scan-tab', children=[
        dcc.Tab(label='CT Scan HU Analysis', value='ct-scan-tab'),
        dcc.Tab(label='Machine Learning - Clustering', value='ml-tab'),
        dcc.Tab(label='LLM Assistant', value='llm-tab')  # New tab for the LLM Assistant
    ]),
    html.Div(id='tabs-content'),
    html.Button('Clear Cache', id='clear-cache-button'),
    html.Div(id='cache-clear-status', style={'margin-top': '10px', 'color': 'green'})
])

# Callback to update the content based on selected tab
@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'ct-scan-tab':
        return ct_scan_layout  
    elif tab == 'ml-tab':
        return ML_layout  
    elif tab == 'llm-tab':
        return html.Div([
            html.H3("LLM Assistant", style={'text-align': 'center'}),
            html.Div([
                dcc.Input(
                    id='llm-prompt', 
                    type='text', 
                    placeholder="Enter your query...",
                    style={'width': '80%', 'padding': '10px', 'margin-right': '10px'}
                ),
                html.Button('Get Response', id='llm-submit-btn', n_clicks=0, style={'padding': '10px'})
            ], style={'display': 'flex', 'justify-content': 'center', 'margin-bottom': '20px'}),
            html.Div(
                id='llm-response', 
                style={'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '5px'}
            )
        ])

# Cache clearing callback
@app.callback(
    [Output('cache-clear-status', 'children'), Output('cache-clear-status', 'style')],
    Input('clear-cache-button', 'n_clicks')
)
def clear_cache(n_clicks):
    if n_clicks:
        cache.clear()
        return "Cache cleared!", {'color': 'green', 'margin-top': '10px'}
    return "", {'display': 'none'}

# Callback to interact with LLM API
@app.callback(
    Output('llm-response', 'children'),
    [Input('llm-submit-btn', 'n_clicks'), Input('llm-prompt', 'value')]
)
def update_llm_response(n_clicks, prompt):
    if n_clicks > 0 and prompt:
        try:
            # Send a request to the FastAPI server
            response = requests.post(LLM_API_URL, json={"prompt": prompt})
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("response", "No response from LLM.")
            else:
                return f"Error: LLM API returned status code {response.status_code}."
        except requests.exceptions.ConnectionError:
            return "Error: Unable to connect to the LLM server. Ensure it is running."
        except Exception as e:
            return f"Error: {e}"
    return "Enter a prompt and click 'Get Response'."

# Register callbacks for the CT Scan Dashboard and ML dashboard
register_callbacks(app)
register_callback_ML(app)

if __name__ == '__main__':
    app.run_server(debug=True)
