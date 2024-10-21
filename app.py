# app.py

import dash
from dash import dcc, html, Input, Output
from ct_scan_dashboard import ct_scan_layout, register_callbacks
from brain_region_dashboard import brain_region_layout, register_callback

# Create the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server 

# Set the layout of the app
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='ct-scan-tab', children=[
        dcc.Tab(label='CT Scan HU Analysis', value='ct-scan-tab'),
        dcc.Tab(label='Zscape analysis Brain', value='brain-region-tab')
    ]),
    html.Div(id='tabs-content')
])

# Callback to update the content based on selected tab
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'ct-scan-tab':
        return ct_scan_layout  # Layout for CT Scan Analysis Dashboard
    elif tab == 'brain-region-tab':
        return brain_region_layout  # Layout for Brain Region Z-Score Analysis


# Register callbacks for the CT Scan Dashboard
register_callbacks(app)
register_callback(app)
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
