#CT Scan Analysis Dashboard

from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import base64
import io
import numpy as np
import re
from sklearn.cluster import KMeans
import scipy.stats as stats


slab_options = [
    'FULL_SCAN', 'L3mid', 'avg-L3mid[3]', 'T1start-to-L5end', 'L1start-to-L1end', 
    'L4start-to-scanend', 'Head', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4',
    'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'SACRUM', 'LSTV', 'Femur', 'Knee', 'Tibia', 'Feet'
]

# Helper function to generate left/right variants for tissues
def generate_tissue_variants(base_name, sides=['L', 'R']):
    return [f"{side}{base_name}" for side in sides]

# Improved tissue definitions with nested structure and programmatic generation
tissue_definitions = {
    'Muscle Group': {
        'Major Muscles': ['ALLSKM','SKM'] + generate_tissue_variants('PECMJR') + generate_tissue_variants('PECMNR'),
        'Head Muscles': generate_tissue_variants('TEMPORALIS') + generate_tissue_variants('MASSETER'),
    },
    'Adipose Tissue': ['VAT', 'EpAT', 'PaAT', 'ThAT', 'SAT', 'LASAT', 'RASAT'],
    'IMAT': {
        'All IMAT': ['ALLIMAT', 'ALLIMAT_NOARMS'],
        'Upper Lower IMAT': generate_tissue_variants('UPlGMAT') + generate_tissue_variants('LWLGIMAT'),
        'Arm IMAT': generate_tissue_variants('AIMAT')
    },
    'Organ Metrics': {
        'Liver/Kidney': ['LIV', 'SPL', 'LKID', 'RKID'],
        'Heart/Pancreas/Brain': ['HRT', 'PANC', 'BRAIN'],
        'Other Organs': ['GB', 'BLADDER', 'PROSTATE', 'AO', 'LLUNG', 'RLUNG', 'TRACH']
    },
    'Fluid Build-up': ['ASCITES', 'LUNG_EFFUSION'],
    'Cardiovascular': ['AOC', 'CAAC', 'AOC-U-CAAC'],
    'Bones': {
        'Full Bone Metrics': ['ALLBONE', 'ALLBONE_NOARMS'],
        'Femur': generate_tissue_variants('FMRHEAD') + generate_tissue_variants('FMRNECK') + generate_tissue_variants('FMRSHAFT'),
        'Hip Bones': generate_tissue_variants('HPBONE'),
        'Arm Bones': generate_tissue_variants('ABONE'),
        'Trabecular Bone': ['TRBCLR']
    }
}
# Function to flatten tissue definitions and return a mapping of tissues to their groups
def flatten_tissue_definitions(tissue_dict):
    tissue_map = {}
    
    for group, value in tissue_dict.items():
        if isinstance(value, dict):  # If it's a nested dictionary, recursively flatten it
            sub_tissue_map = flatten_tissue_definitions(value)  # Recursive call for sub-dictionaries
            for tissue, subgroup in sub_tissue_map.items():
                tissue_map[tissue] = f"{group} - {subgroup}"  # Include both group and subgroup in the mapping
        else:  # If it's a list, map each tissue to the main group
            for tissue in value:
                tissue_map[tissue] = group
    
    return tissue_map

tissue_map = flatten_tissue_definitions(tissue_definitions)

def classify_tissue(tissue_name):
    for tissue in tissue_map:
        if tissue in tissue_name:  # Check if the tissue name contains any tissue in the map
            return tissue_map[tissue]
    return 'Other' 

# Define color mapping for tissue groups
tissue_color_map = {
    'Muscle Group': 'blue',
    'Adipose Tissue': 'green',
    'IMAT': 'red',
    'Organ Metrics': 'purple',
    'Fluid Build-up': 'orange',
    'Cardiovascular': 'brown',
    'Bones': 'gray',
    'Other': 'black'  # For any other tissues
}

# Global Parsing Function (already implemented)
def parse_column_name(col_name):
    parts = col_name.split(';')
    
    if len(parts) == 1:
        metric = parts[0].strip()
        return None, None, metric  # No tissue, no slab, only metric
    
    elif len(parts) == 3:
        # slab; tissue_with_hu; metric
        slab = parts[0].strip()
        tissue_with_hu = parts[1].strip()
        metric = parts[2].strip()

        # Extract tissue name(s) and HU range using regex
        union_pattern = r'tissue\[\d+,\d+\]-U-tissue\[\d+,\d+\]'
        single_pattern = r'tissue\[\d+,\d+\]'

        match = re.search(union_pattern, tissue_with_hu)
        if match:
            tissue = match.group(0)  # Return the full union pattern
        else:
            match = re.search(single_pattern, tissue_with_hu)
            if match:
                tissue = match.group(0)  # Return the single tissue-HU pattern
            else:
                tissue = re.sub(r'\[.*?\]|-U-.*', '', tissue_with_hu).strip()

        return slab, tissue, metric

    elif len(parts) == 2:
        # slab; metric (no tissue)
        slab = parts[0].strip()
        metric = parts[1].strip()
        return slab, None, metric  # No tissue in this case

    return None, None, None

def classify_vertebrae(slab):
        if any(c in slab for c in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']):
            return 'Cervical'
        elif any(t in slab for t in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']):
            return 'Thoracic'
        elif any(l in slab for l in ['L1', 'L2', 'L3', 'L4', 'L5']):
            return 'Lumbar'
        elif 'SACRUM' in slab or 'LSTV' in slab:
            return 'Sacral'
        return 'Other'  # Default to "Other" if not matched

# Layout for the CT Scan Analysis Dashboard
ct_scan_layout = html.Div([
    html.H1("CT Scan HU Value Analysis Dashboard", style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
               'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'}
    ),
    html.Div(id='file-info', style={'margin-top': '20px'}),
    dcc.Tabs([
        dcc.Tab(label='Heatmap & Distribution', children=[
            html.Div([
                html.Label('Select Tissue:', style={'font-weight': 'bold'}),
                dcc.Dropdown(id='tissue-dropdown', multi=True, style={'width': '100%'}),
                html.Label('Select Slab:', style={'font-weight': 'bold', 'margin-top': '20px'}),
                dcc.Dropdown(id='slab-dropdown', multi=True, style={'width': '100%'}),
                html.Label('Select Measurement Metric:', style={'font-weight': 'bold', 'margin-top': '20px'}),
                dcc.Dropdown(id='measurement-dropdown', multi=True, style={'width': '100%'}),
                html.Label('HU Value Threshold Range:', style={'font-weight': 'bold'}),
                dcc.RangeSlider(
                    id='hu-threshold-slider',
                    min=-32768, max=32768, step=1000,
                    value=[-1000, 3000],
                    marks={-32768: '-32768', -3000: '-3000', -1000: '-1000', 0: '0', 1000: '1000', 3000: '3000', 32768: 'Max'}
                ),
                
            ], style={'margin': '20px'}),

            # Heatmap display area
            dcc.Graph(id='hu-heatmap', style={'margin-top': '30px'}),

            # Distribution plot section
            html.Div([
                html.H3("HU Distribution Patterns", style={'text-align': 'center'}),
                dcc.Graph(id='hu-distribution-plot')
            ], style={'margin-top': '30px'})
        ]),
        dcc.Tab(label='Time-Series Analysis', children=[
            html.Div([
                html.H3("Time-Series Analysis of HU Values", style={'text-align': 'center'}),
                dcc.Graph(id='time-series-plot')
            ])
        ]),
        dcc.Tab(label='Z-Score', children=[
            html.Div([
                html.H3("Z-Score", style={'text-align': 'center'}),
                dcc.Graph(id='z-score-heatmap')
            ])
        ]),
    
        dcc.Tab(label='Clustering & Segmentation', children=[
            html.Div([
                html.H3("Patient Segmentation & Clustering", style={'text-align': 'center'}),
                dcc.Graph(id='clustering-plot')
            ])
        ]),
    
        dcc.Tab(label='Vertebrae Count', children=[
            html.Div([
               html.H3("Vertebrae Count", style={'text-align': 'center'}),
                dcc.Graph(id='vertebrae-count-plot')
                ])
        ]),
        dcc.Tab(label='Tissue Count Plot', children=[
            html.Div([
               html.H3("Tissue Count Plot", style={'text-align': 'center'}),
               dcc.Graph(id='tissue-count-plot')
               ])
        ]),
         dcc.Tab(label='Slices per Patient', children=[
            html.Div([
               html.H3("Slices per Patient", style={'text-align': 'center'}),
               dcc.Graph(id='gender-slices-plot')
               ])
        ]),
      
    ])
])


# Helper functions to parse the uploaded files
def parse_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if filename.endswith('.csv'):
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        return None
    
    return df

def parse_column_name(col_name):
    # Split the column name by semicolon
    parts = col_name.split(';')
    
    # If there are no separators, treat the column name as a metric
    if len(parts) == 1:
        # Treat this as a standalone metric (like PatientID, Gender, etc.)
        metric = parts[0].strip()
        return None, None, metric  # No tissue, no slab, only metric
    
    elif len(parts) == 3:
        # slab; tissue_with_hu; metric
        slab = parts[0].strip()
        tissue_with_hu = parts[1].strip()
        metric = parts[2].strip()

        # Extract tissue name(s) and HU range using regex
        union_pattern = r'tissue\[\d+,\d+\]-U-tissue\[\d+,\d+\]'
        single_pattern = r'tissue\[\d+,\d+\]'

        # Try to match the union pattern first
        match = re.search(union_pattern, tissue_with_hu)
        if match:
            tissue = match.group(0)  # Return the full union pattern
        else:
            # If no union pattern found, fall back to single HU range or simple tissue name
            match = re.search(single_pattern, tissue_with_hu)
            if match:
                tissue = match.group(0)  # Return the single tissue-HU pattern
            else:
                # Fallback to just the tissue name by removing any HU ranges or union patterns
                tissue = re.sub(r'\[.*?\]|-U-.*', '', tissue_with_hu).strip()

        return slab, tissue, metric

    elif len(parts) == 2:
        # slab; metric (no tissue)
        slab = parts[0].strip()
        metric = parts[1].strip()
        return slab, None, metric  # No tissue in this case

    return None, None, None

# Register the callbacks for updating dropdowns and generating various graphs
def register_callbacks(app):

    # Update dropdowns based on the uploaded file
    @app.callback(
    [Output('tissue-dropdown', 'options'),
     Output('tissue-dropdown', 'value'),
     Output('slab-dropdown', 'options'),
     Output('slab-dropdown', 'value'),
     Output('measurement-dropdown', 'options'),
     Output('measurement-dropdown', 'value')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')]
)
    def update_dropdowns(contents, filename):
        if contents is None:
            return [], [], [],[], [], []

        df = parse_file(contents, filename)
        if df is None:
            return [], [], [],[], [], []

        slab_set = set()
        tissue_set = set()
        metric_set = set()

        for col in df.columns:
            slab, tissue, metric = parse_column_name(col)
            if slab:
                slab_set.add(slab)
            if tissue:
                tissue_set.add(tissue)
            if metric:
                metric_set.add(metric)

        # Convert sets to lists for options and values
        slab_options = [{'label': slab, 'value': slab} for slab in slab_set]
        tissue_options = [{'label': tissue, 'value': tissue} for tissue in tissue_set]
        measurement_metric_options = [{'label': metric, 'value': metric} for metric in metric_set]

        # Select all options by default
        slab_values = [slab['value'] for slab in slab_options]
        tissue_values = [tissue['value'] for tissue in tissue_options] if tissue_options else []
        measurement_metric_values = [metric['value'] for metric in measurement_metric_options]

        return tissue_options, tissue_values, slab_options, slab_values, measurement_metric_options, measurement_metric_values


    # Heatmap & Distribution Plot callback
    @app.callback(
    [Output('hu-heatmap', 'figure'), Output('hu-distribution-plot', 'figure')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('tissue-dropdown', 'value'),
     Input('slab-dropdown', 'value'),
     Input('measurement-dropdown', 'value'),
     Input('hu-threshold-slider', 'value')])  
    def update_heatmap_and_distribution(contents, filename, selected_tissues, selected_slabs, selected_metrics, hu_threshold):
        if contents is None:
            return go.Figure(), go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure(), go.Figure()

        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues]) if selected_tissues else ''
        slab_filter = '|'.join([re.escape(slab) for slab in selected_slabs]) if selected_slabs else ''
        metric_filter = '|'.join([re.escape(metric) for metric in selected_metrics]) if selected_metrics else ''

        filtered_columns = df.columns[
            df.columns.str.contains(tissue_filter) &
            df.columns.str.contains(slab_filter) &
            df.columns.str.contains(metric_filter)
        ]

        filtered_data = df[filtered_columns]

        # Apply HU threshold range filtering
        hu_min, hu_max = hu_threshold
        filtered_data = filtered_data[(filtered_data >= hu_min) & (filtered_data <= hu_max)]

        if filtered_data.empty:
            return go.Figure(), go.Figure()

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=filtered_data.iloc[:, 0],
            x=filtered_columns.values,
            y=filtered_data.index,
            colorscale='Jet',
            colorbar=dict(title="HU Value"),
        ))
        heatmap_fig.update_layout(autosize=False, margin=dict(l=20, r=20, t=50, b=50), height=600, width=1400)

        hist_fig = px.histogram(filtered_data, x=filtered_data.columns[0], nbins=50, title='HU Distribution',
                                marginal='rug', opacity=0.7, color_discrete_sequence=['blue'])
        hist_fig.update_layout(xaxis_title='HU Value', yaxis_title='Frequency',
                            autosize=False, margin=dict(l=20, r=20, t=50, b=50), height=600, width=1400)

        return heatmap_fig, hist_fig

    # Time-Series Analysis callback
    @app.callback(
        Output('time-series-plot', 'figure'),
        [Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('tissue-dropdown', 'value')] )
    def update_time_series_plot(contents, filename, selected_tissues):
        if contents is None or not selected_tissues:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()
        # Check if 'Timestamp' exists
        if 'Timestamp' not in df.columns:
            return go.Figure()

        # Convert 'Timestamp' column to datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # Ensure selected_tissues is a list
        if not isinstance(selected_tissues, list):
            selected_tissues = [selected_tissues]

        # Filter columns that match the selected tissues
        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues])
        filtered_columns = df.columns[df.columns.str.contains(tissue_filter, regex=True)]

        # Ensure we have columns to plot
        if filtered_columns.empty:
            return go.Figure()
    

        # Create a time-series plot
        filtered_data = df[['Timestamp'] + list(filtered_columns)]
        time_series_fig = px.line(
            filtered_data, x='Timestamp', y=filtered_columns[0],
            title='Time-Series Analysis of HU Values'
        )
        time_series_fig.update_layout(xaxis_title='Time', yaxis_title='HU Value',
    autosize=False,  # Disable autosizing to manually control
    margin=dict(l=20, r=20, t=50, b=50),  # Tighter margins
    height=600,  # Adjust height as needed
    width=1400,   # Adjust width as needed
    )

        return time_series_fig

    @app.callback(
    Output('z-score-heatmap', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('tissue-dropdown', 'value'),
     Input('hu-threshold-slider', 'value')])  
    def update_z_score(contents, filename, selected_tissues, hu_threshold):
        if contents is None or not selected_tissues:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()
        df = df.infer_objects(copy=False)
    
    # Select numeric columns for interpolation
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear', inplace=False)
    

        # Ensure selected_tissues is a list
        if not isinstance(selected_tissues, list):
            selected_tissues = [selected_tissues]

        # Filter columns that match the selected tissues
        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues])
        filtered_columns = df.columns[df.columns.str.contains(tissue_filter, regex=True)]

        if len(filtered_columns) == 0:
            return go.Figure()

        # Z-Score calculation
        filtered_data = df[filtered_columns]
        patient_ids = df['PatientID'].values

        # Apply HU threshold range filtering
        hu_min, hu_max = hu_threshold
        filtered_data = filtered_data[(filtered_data >= hu_min) & (filtered_data <= hu_max)]

        z_scores = (filtered_data - filtered_data.mean()) / filtered_data.std()
        z_scores_clipped = z_scores.clip(lower=-5, upper=5)

        z_threshold = 0
        z_scores_highlighted = z_scores_clipped.map(lambda x: x if abs(x) > z_threshold else None)
        tissue_names = [parse_column_name(col)[1] for col in filtered_columns]
        # Create Z-Score heatmap
        z_score_fig = px.imshow(
            z_scores_highlighted.T, 
            labels={ 'x': 'Patient ID',   # X-axis label for patients
        'y': 'Tissue Type',               # Y-axis label for tissue types
        'z': 'Z-Score',
         },
            title='Z-Score Heatmap Grouped by Tissue Type',
            x= patient_ids,
            color_continuous_scale='RdBu',
            aspect="auto"
        )

        z_score_fig.update_yaxes(
            tickvals=list(range(len(filtered_columns))),
            ticktext=tissue_names,
            tickfont=dict(color='black')  
        )
        z_score_fig.update_traces(
        hovertemplate=(
        "<b>Patient ID:</b> %{x}<br>" +
        "<b>Tissue Type:</b> %{y}<br>" +
        "<b>Z-Score:</b> %{z:.2f}<br>" 
        )
)

        z_score_fig.update_layout(
            xaxis_title='Patient',
            yaxis_title='Tissue',
            autosize=False,
            margin=dict(l=20, r=20, t=50, b=50),
            height=600,
            width=1400
        )

        return z_score_fig

    # Clustering and Segmentation callback
    @app.callback(
    Output('clustering-plot', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('hu-threshold-slider', 'value')])  
    def update_clustering_plot(contents, filename, hu_threshold):
        if contents is None:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()
        
        ddf = df.infer_objects(copy=False)
    
        # Select numeric columns for interpolation
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear', inplace=False)
    

        # Automatically detect the column containing 'HU' in the name
        hu_columns = [col for col in df.columns if 'HU' in col]

        if len(hu_columns) == 0:
            return go.Figure()  # Return an empty figure if no HU columns found

        # Use the first HU-related column for clustering
        selected_hu_column = hu_columns[0]

        # Apply HU threshold range filtering
        hu_min, hu_max = hu_threshold
        df = df[(df[selected_hu_column] >= hu_min) & (df[selected_hu_column] <= hu_max)]

        # Perform KMeans clustering on the HU data
        kmeans = KMeans(n_clusters=3).fit(df[[selected_hu_column]])
        df['Cluster'] = kmeans.labels_

        # Create the clustering plot
        cluster_fig = px.scatter(df, x=df.index, y=selected_hu_column, color='Cluster',
                                title='Patient Segmentation & Clustering')
        cluster_fig.update_layout(
            autosize=False,
            margin=dict(l=20, r=20, t=50, b=50),
            height=600,
            width=1400
        )

        return cluster_fig


    @app.callback(
    Output('vertebrae-count-plot', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('slab-dropdown', 'value'),   Input('measurement-dropdown', 'value')] )
    def update_vertebrae_count_plot(contents, filename, selected_scans, selected_metrics):
        if contents is None:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()

        # Filter based on selected scan types
        scan_filter = '|'.join([re.escape(scan) for scan in selected_scans]) if selected_scans else ''
        metric_filter = '|'.join([re.escape(metric) for metric in selected_metrics]) if selected_metrics else ''

        filtered_columns = df.columns[
            df.columns.str.contains(scan_filter)&
            df.columns.str.contains(metric_filter)
        ]

        slab_names = [parse_column_name(col)[0] for col in filtered_columns]
       

        if filtered_columns.empty:
            return go.Figure()

        # Count the occurrences of each scan type
        vertebrae_count = df[filtered_columns].count()

        # Classify each slab into vertebra regions
        vertebra_regions = [classify_vertebrae(slab) for slab in slab_names]
        metric= [parse_column_name(col)[2] for col in filtered_columns]

        data = pd.DataFrame({
        'Metric': metric,  
    })


        # Create a bar chart with vertebra count, color-coded by vertebra region
        vertebrae_fig = px.bar(
            data_frame=data,
            x=slab_names,
            y=vertebrae_count.values,
            hover_data={
                'Metric': True,
            },
            color=vertebra_regions,
            title='Vertebra Count Based on Scan Types',
            labels={
            'x': 'Vertebrae Type',
            'y': 'Number of Scans',
            'color': 'Vertebra Region', 
            'metric':'Metric'

        },
            color_discrete_map={
                'Cervical': 'blue',
                'Thoracic': 'green',
                'Lumbar': 'red',
                'Sacral': 'orange',
                'Other': 'gray'
            }
        )

        # Sort the x-axis by the count of vertebrae for better readability
        vertebrae_fig.update_layout(
            xaxis_title='Vertebra Type',
            yaxis_title='Count of Scans',
            autosize=False,
            margin=dict(l=20, r=20, t=50, b=50),
            height=600,
            width=1400,
            xaxis={'categoryorder': 'total descending'},
            hovermode='x'
        )

        return vertebrae_fig

    
    @app.callback(
        Output('tissue-count-plot', 'figure'),
        [Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('tissue-dropdown', 'value'),
        Input('slab-dropdown', 'value'),
        Input('measurement-dropdown', 'value')])  # Assuming you're using a tissue dropdown
    def update_tissue_count_plot(contents, filename, selected_tissues, selected_slabs, selected_metrics):
        if contents is None:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()


        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues]) if selected_tissues else ''
        slab_filter = '|'.join([re.escape(slab) for slab in selected_slabs]) if selected_slabs else ''
        metric_filter = '|'.join([re.escape(metric) for metric in selected_metrics]) if selected_metrics else ''
        

        filtered_columns = df.columns[
            df.columns.str.contains(tissue_filter) &
            df.columns.str.contains(slab_filter)&
            df.columns.str.contains(metric_filter)
        ]

        slab_names = [parse_column_name(col)[0] for col in filtered_columns]

        tissue_names = [parse_column_name(col)[1] for col in filtered_columns]  
        
        metric = [parse_column_name(col)[2] for col in filtered_columns]

        if filtered_columns.empty:
            return go.Figure()

        # Count the occurrences of each tissue
        tissue_count = df[filtered_columns].count()

        # Classify tissues into groups
        tissue_groups = [classify_tissue(tissue) for tissue in tissue_names]
        
        # Creating the DataFrame with relevant fields
        data = pd.DataFrame({
        'Tissue Name': tissue_names,
        'Tissue Sample Count': tissue_count.values,
        'Tissue Group': tissue_groups,
        'Metric': metric,  
        'Scan Region': slab_names,  
    })

        # Create the bar chart
        tissue_groups_fig = px.bar(
            data_frame=data,
            x='Tissue Name',
            y='Tissue Sample Count',
            color='Tissue Group',
            hover_data={
                'Metric': True,
                'Scan Region': True,
            },
            title='Tissue Count Based on Selected Tissues',
            labels={
                'Tissue Name': 'Tissue Name',
                'Tissue Sample Count': 'Tissue Sample Count',
                'Tissue Group': 'Tissue Group',
                'Metric': 'Metric',
                'Scan Region': 'Scan Region',
            },
            color_discrete_map=tissue_color_map
        )
        return tissue_groups_fig
    
    @app.callback(
    Output('gender-slices-plot', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
    def update_gender_slices_plot(contents, filename):
        if contents is None:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()

        # Find all columns that contain 'num_slices' in their names
        num_slices_cols = [col for col in df.columns if 'num_slices' in col]
        
        # Find slice thickness column (in your case it's 'height_mm')
        height_mm_cols = [col for col in df.columns if 'height_mm' in col]

        # If no such columns are found, return an empty figure
        if len(num_slices_cols) == 0:
            print("No column found with 'num_slices' in its name.")
            return go.Figure()

        # Convert all num_slices and height_mm columns to numeric, filling NaNs with 0
        df[num_slices_cols] = df[num_slices_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df[height_mm_cols] = df[height_mm_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Ensure 'Gender', 'PatientID', and other required columns exist
        if 'Gender' not in df.columns or 'PatientID' not in df.columns:
            print("Missing 'Gender' or 'PatientID' columns.")
            return go.Figure()

        # For each slice, we want to have an individual record
        slice_data = pd.DataFrame()

        # Iterate through each slice column
        for slice_col in num_slices_cols:
            # Create a new dataframe with individual slice info
            slice_info = df[['Gender', 'PatientID']].copy()
            slice_info['num_slices'] = df[slice_col]
            
            # If we have height_mm (slice thickness), include it
            if len(height_mm_cols) > 0:
                slice_info['height_mm'] = df[height_mm_cols[0]]
                slice_info['total_thickness'] = slice_info['num_slices'] * slice_info['height_mm']
            else:
                slice_info['height_mm'] = np.nan
                slice_info['total_thickness'] = np.nan

            # Add a column for the specific slice type (e.g., L3, L4, etc.)
            slice_info['slice_type'] = slice_col
            
            # Append this slice's data to the overall slice data
            slice_data = pd.concat([slice_data, slice_info])

        # Sort by Patient ID and Gender for better visualization
        slice_data = slice_data.sort_values(by=['PatientID', 'Gender'])

        # Assuming 'Gender' and other fields are available
        slice_data_fig = px.bar(
            data_frame=slice_data,
            x='slice_type',  # Use slice_type as the Slab
            y='height_mm',  # Anatomical height or total thickness
            color='Gender', 
            hover_data=['num_slices', 'total_thickness', 'Gender', 'PatientID'],  # Include Gender in hover data
            title='Slice Thickness and Volume per Slab',
            labels={
                'height_mm': 'Slice Thickness (mm)',
                'num_slices': 'Number of Slices',
                'total_thickness': 'Total Thickness (mm)',
                'Gender': 'Gender',
                'PatientID':'Patient ID',
                'slice_type': 'Slab'
            },
            barmode='group'
        )

        # Add custom hover info to show slice thickness, number of slices, gender, etc.
        slice_data_fig.update_traces(
            hovertemplate="<b>Slab:</b> %{x}<br>" +
                        "<b>Slice Thickness:</b> %{y} mm<br>" +
                        "<b>Number of Slices:</b> %{customdata[0]}<br>" +
                        "<b>Total Thickness:</b> %{customdata[1]} mm<br>" +
                        "<b>Patient ID:</b> %{customdata[3]}<br>"+
                        "<b>Gender:</b> %{customdata[2]}"
        )

        slice_data_fig.update_layout(
            xaxis_title='Slab',
            yaxis_title='Total Thickness (mm)',
            autosize=False,
            height=600,
            width=1400
        )

        return slice_data_fig
