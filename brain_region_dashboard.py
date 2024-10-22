#CT Scan Analysis Dashboard

from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

# Define the brain regions and patient groups
regions = [
    'accumbens-area', 'rostralanteriorcingulate', 'caudalanteriorcingulate',
    'medialorbitofrontal', 'frontalpole', 'lateralorbitofrontal', 'parsorbitalis',
    'rostralmiddlefrontal', 'parstriangularis', 'caudalmiddlefrontal', 'superiorfrontal',
    'parsopercularis', 'precentral', 'postcentral', 'paracentral', 'posteriorcingulate',
    'isthmuscingulate', 'precuneus', 'superiorparietal', 'inferiorparietal', 'supramarginal',
    'bankssts', 'insula', 'superiortemporal', 'transversetemporal', 'middletemporal', 
    'inferiortemporal', 'lateraloccipital', 'cuneus', 'pericalcarine', 'lingual', 'fusiform', 
    'parahippocampal', 'entorhinal', 'temporalpole', 'hippocampus', 'amygdala', 'thalamus', 
    'pallidum', 'putamen', 'caudate', 'cerebellum-cortex', 'lateral-ventricle', 'brainstem'
]
patients = ['SNC', 'uNC', 'SMCI', 'PNC', 'PMCI', 'eDAT', 'SDAT']

# Generate random Z-score data for demonstration purposes
np.random.seed(900)  # For reproducibility of the random data
z_scores_data = np.random.uniform(-5, 5, (len(regions), len(patients), 5000))  # Data for 5000 different time points

# Define the custom color scale for the heatmap
custom_colorscale = [
    [0.0, 'rgb(0, 0, 255)'],       # Dark blue at the lowest value
    [0.2, 'rgb(0, 128, 255)'],     # Medium blue
    [0.3, 'rgb(0, 204, 255)'],     # Light blue
    [0.4, 'rgb(0, 255, 255)'],     # Cyan
    [0.45, 'rgb(0, 255, 128)'],    # Light green
    [0.48, 'rgb(0, 255, 0)'],      # Green
    [0.49, 'rgb(128, 255, 0)'],    # Yellow-green
    [0.5, 'rgb(255, 255, 255)'],   # White at zero
    [0.51, 'rgb(255, 204, 0)'],    # Yellowish-orange
    [0.55, 'rgb(255, 153, 0)'],    # Light orange
    [0.6, 'rgb(255, 102, 0)'],     # Orange-red
    [0.85, 'rgb(255, 51, 0)'],     # Light red
    [1.0, 'rgb(255, 0, 0)']        # Red at the highest value
]

# Layout for Brain Region Z-Score Analysis Dashboard
brain_region_layout = html.Div([
    html.H1("Brain Region Z-Score Analysis", style={'text-align': 'center'}),

    # Dropdown for graph type selection
    html.Div([
        html.Label('Select Graph Type:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='graph-type-dropdown',
            options=[
                {'label': 'Heatmap', 'value': 'heatmap'},
                {'label': '3D Surface Plot', 'value': '3d-surface'}
            ],
            value='heatmap',  # Default value
            style={'width': '100%', 'margin-bottom': '20px'}
        ),
    ]),

    # Interactive controls for the heatmap
    html.Div(id='interactive-controls', children=[
        html.Label('Select Brain Region:'),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': region, 'value': region} for region in regions],
            value=regions,  # Default to showing all regions
            multi=True,
            style={'width': '100%'}
        ),
        html.Label('Select Patient Groups:'),
        dcc.Dropdown(
            id='patient-dropdown',
            options=[{'label': patient, 'value': patient} for patient in patients],
            value=patients,  # Default to showing all patient groups
            multi=True,
            style={'width': '100%'}
        ),
        html.Label('Select Timepoint or Subgroup:'),
        dcc.Slider(
            id='time-slider',
            min=0,
            max=4999,
            step=1,
            value=0,
            marks={i: f'{i+1}' for i in range(0, 5000, 1000)},
        ),
        html.Label('Set Z-Score Threshold Range:'),
        dcc.RangeSlider(
            id='zscore-threshold-slider',
            min=-5,
            max=5,
            step=0.1,
            value=[-5, 5],  # Default to the full range
            marks={-5: '-5', -2: '-2', 0: '0', 2: '2', 5: '5'},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'display': 'block'}),  # Show controls only for heatmap

    # Graph display area
    dcc.Graph(id='dynamic-graph')  # Graph component for dynamic rendering
], style={'padding': '20px'})

# Callback to update the graph based on the dropdown selection and interactive controls
def register_callback(app):
    @app.callback(
        [Output('dynamic-graph', 'figure'),
         Output('interactive-controls', 'style')],
        [Input('graph-type-dropdown', 'value'),
         Input('region-dropdown', 'value'),
         Input('patient-dropdown', 'value'),
         Input('time-slider', 'value'),
         Input('zscore-threshold-slider', 'value')]
    )
    def update_graph(graph_type, selected_regions, selected_patients, selected_time, zscore_threshold):
        # Ensure the data is properly filtered for both heatmap and 3D surface plot
        region_indices = [regions.index(region) for region in selected_regions]
        patient_indices = [patients.index(patient) for patient in selected_patients]
        z_data = z_scores_data[np.ix_(region_indices, patient_indices, [selected_time])].squeeze()

        # Apply thresholding to the data for both visualizations
        z_data_filtered = np.where((z_data >= zscore_threshold[0]) & (z_data <= zscore_threshold[1]), z_data, np.nan)

        if graph_type == 'heatmap':
            # Create the heatmap figure
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=z_data_filtered,
                x=[patients[i] for i in patient_indices],
                y=[regions[i] for i in region_indices],
                colorscale=custom_colorscale,
                zmin=-5, zmax=5,
                colorbar=dict(title="Z-Score"),
                hovertemplate="Region: %{y}<br>Patient: %{x}<br>Z-Score: %{z}<extra></extra>"
            ))

            heatmap_fig.update_layout(
                title=f"Z-Score Heatmap for Timepoint {selected_time + 1}",
                xaxis_title="Patient Groups",
                yaxis_title="Brain Regions"
            )

            # Return the heatmap figure and display style for controls
            return heatmap_fig, {'display': 'block'}
        else:
            # Create the 3D surface plot figure
            fig_surface = go.Figure(data=[go.Surface(
                z=z_data_filtered,
                x=[patients[i] for i in patient_indices],
                y=[regions[i] for i in region_indices],
                colorscale='Viridis',
                hoverinfo='x+y+z', 
            )])

            fig_surface.update_layout(
                scene=dict(
                    xaxis_title='Patients/Conditions',
                    yaxis_title='Brain Regions',
                    zaxis_title='Z-Scores',
                    aspectmode='manual',
                    aspectratio=dict(x=3, y=2, z=1),
                ),
                scene_camera=dict(eye=dict(x=3, y=3, z=2)),
                margin=dict(l=20, r=20, b=50, t=50),
                width=1200,
                height=800
            )

            # Return the 3D surface plot figure and display style for controls
            return fig_surface, {'display': 'block'}
