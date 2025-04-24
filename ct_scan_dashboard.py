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
from plotly.subplots import make_subplots
from plotly.graph_objs import Histogram, Scatter
from dash.dash_table import DataTable


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
    'Other': 'black'  
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
    
    else:
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
        return 'Other' 

# Layout for the CT Scan Analysis Dashboard
ct_scan_layout = html.Div([
    html.H1("CT Scan HU Value Analysis Dashboard", style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
               'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'}
    ),    

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
                    value=[-32768, 32768],
                    marks={-32768: '-32768', -3000: '-3000', -1000: '-1000', 0: '0', 1000: '1000', 3000: '3000', 32768: 'Max'}
                ),
                html.Label('Select Patient:'),
                dcc.Dropdown(
                    id='patient-dropdown',
                    placeholder="Select a patient ID"
                    ),
                html.Label('Select HU Metric:'),
                dcc.RadioItems(
                     id='hu-metric-radio',
                       options=[
                           {'label': 'HU Mean', 'value': 'HU_mean'},
                           {'label': 'HU Max', 'value': 'HU_max'},
                           {'label': 'HU Std', 'value': 'HU_std'},
                           {'label': 'HU Median', 'value': 'HU_median'},
                           {'label': 'HU Min', 'value': 'HU_min'}
            ],
            value='HU_mean', 
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
                
            ], style={'margin': '20px'}),

    html.Div(id='file-info', style={'margin-top': '20px'}),
    dcc.Tabs([
        
        dcc.Tab(label='Data Preview', children=[
            # Data Preview Area
            html.Div([
                html.H3("Uploaded Data Preview", style={'text-align': 'center'}),
                DataTable(
                    id='data-preview-table',
                    style_table={'overflowX': 'auto', 'margin-top': '20px'},
                    style_cell={'textAlign': 'left', 'padding': '10px', 'minWidth': '50px', 'width': '100px', 'maxWidth': '200px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                    
                    
        )
    ]),

        ]),
        dcc.Tab(label='HU distribution and Time-Series Analysis', children=[
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
         dcc.Tab(label='Slices thickness and Gender', children=[
            html.Div([
               html.H3("Avg Slice thickenss for scan types per patient", style={'text-align': 'center'}),
               dcc.Graph(id='gender-slices-plot'),
               dcc.Graph(id='gender-slices-line')      
    ])
               ])
        ]),
      
    ])


def parse_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    if filename.endswith('.csv'):
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        return None  # Unsupported file format
    df.replace('NA', pd.NA, inplace=True)
    df = df.dropna(axis=1, how='any')
    
    return df.to_dict('records')


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
    [Input('ct-scan-data-store', 'data')]
)
    def update_dropdowns(data):
        if data is None:
            return [], [], [],[], [], []

        df = pd.DataFrame(data)

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
    
    @app.callback(
        Output('ct-scan-data-store', 'data'),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
    )
    def load_and_store_data(contents, filename):
        if contents is None:
            return None
        
        # Use parse_file to handle data extraction
        parsed_data = parse_file(contents, filename)
        return parsed_data 
   
    @app.callback(
    Output('data-preview-table', 'data'),
    Output('data-preview-table', 'columns'),
    Input('ct-scan-data-store', 'data')
    )
    def update_data_preview(data):
        if data is None:
            return [], []

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Create a list of column definitions for the DataTable
        columns = [{'name': col, 'id': col} for col in df.columns]

        # Return the first 10 rows of the DataFrame as a list of dictionaries
        return df.head(10).to_dict('records'), columns


    # Heatmap & Distribution Plot callback
    @app.callback(
        Output('hu-distribution-plot', 'figure'),
        [Input('ct-scan-data-store', 'data'),
        Input('tissue-dropdown', 'value'),
        Input('slab-dropdown', 'value'),
        Input('measurement-dropdown', 'value'), 
        Input('hu-metric-radio', 'value'),
        Input('hu-threshold-slider', 'value')])  
    def update_heatmap_and_distribution(data, selected_tissues, selected_slabs, selected_metrics, HU_selected_metric,hu_threshold):
        if data is None:
            return go.Figure()

        df = pd.DataFrame(data)
        
        # Generate the filters for tissues, slabs, and metrics
        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues]) if selected_tissues else ''
        slab_filter = '|'.join([re.escape(slab) for slab in selected_slabs]) if selected_slabs else ''
        HU_metric_filter = re.escape(HU_selected_metric)

        # Apply the combined filter to find matching columns
        filtered_columns = df.columns[
            df.columns.str.contains(tissue_filter) &
            df.columns.str.contains(slab_filter) &
            df.columns.str.contains(HU_metric_filter)
        ]

        if len(filtered_columns) == 0:
            return go.Figure()

        # Extract relevant data from the filtered columns
        
        filtered_data = df[filtered_columns].apply(pd.to_numeric, errors='coerce').astype(float)

        hu_values = filtered_data.values.flatten()
        hu_values = hu_values[~pd.isna(hu_values)]  # Remove NaN values (if any remain)


        hist_fig = px.histogram(
            x=hu_values,
            nbins=50,
            title=f'{HU_selected_metric} Distribution',
            labels={'x': f'{HU_selected_metric}', 'y': 'Frequency'},
            marginal='rug',
            opacity=0.7,
            color_discrete_sequence=['blue']
        )

        hist_fig.update_layout(
            xaxis_title=f'{HU_selected_metric}',
            yaxis_title='Frequency',
            autosize=False,
            margin=dict(l=20, r=20, t=50, b=50),
            height=800,
            width=1400
        )

        return hist_fig
    # Time-Series Analysis callback
    @app.callback(
    Output('time-series-plot', 'figure'),
    [
        Input('ct-scan-data-store', 'data'),
        Input('tissue-dropdown', 'value'),
        Input('slab-dropdown', 'value'),
        Input('hu-metric-radio', 'value')
    ]
)
    def update_time_series_plot(data, selected_tissues, selected_slabs, HU_selected_metric):
        if not data or not selected_tissues or not HU_selected_metric:
            return go.Figure()

        # Load data into DataFrame
        df = pd.DataFrame(data)

        # Ensure required columns exist
        if 'PatientID' not in df.columns or 'SeriesDate' not in df.columns:
            print("Missing required columns.")
            return go.Figure()

        # Convert SeriesDate to datetime and sort by PatientID and SeriesDate
        df['SeriesDate'] = pd.to_datetime(df['SeriesDate'], format='%Y%m%d', errors='coerce')
        df.sort_values(by=['PatientID', 'SeriesDate'], inplace=True)

        # Filter relevant columns based on dropdown selections
        column_filter = (
            df.columns.str.contains('|'.join(selected_tissues)) &
            df.columns.str.contains('|'.join(selected_slabs)) &
            df.columns.str.contains(HU_selected_metric)
        )
        filtered_columns = df.columns[column_filter]

        if filtered_columns.empty:
            print("No matching columns for the selected filters.")
            return go.Figure()

        # Extract and flatten HU values for the histogram and time-series
        filtered_data = df[filtered_columns].apply(pd.to_numeric, errors='coerce').astype(float)
        hu_values = filtered_data.values.flatten()
        hu_values = hu_values[~pd.isna(hu_values)]  # Remove NaN values

        # Align HU values with PatientID and SeriesDate
        expanded_df = pd.DataFrame({
            'HU_Value': hu_values,
            'SeriesDate': df['SeriesDate'].repeat(filtered_columns.shape[0]).reset_index(drop=True),
            'PatientID': df['PatientID'].repeat(filtered_columns.shape[0]).reset_index(drop=True),
        })



        # Create the histogram
        hist_fig = px.histogram(
            x=hu_values,
            nbins=50,
            title=f'{HU_selected_metric} Distribution',
            labels={'x': f'{HU_selected_metric}', 'y': 'Frequency'},
            marginal='rug',
            opacity=0.7,
            color_discrete_sequence=['blue']
        )

        # Create the time-series plot
        time_series_fig = px.line(
            expanded_df,
            x='SeriesDate',
            y='HU_Value',
            color='PatientID',
            line_group='PatientID',
            title='Time-Series Analysis of HU Values',
            labels={'HU_Value': f'{HU_selected_metric} (HU)', 'SeriesDate': 'Date', 'PatientID': 'Patient ID'}
        )

        combined_fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{HU_selected_metric} Distribution', 'Time-Series Analysis'),
            shared_xaxes=False
        )

        # Add histogram
        combined_fig.add_trace(
            Histogram(x=hu_values, nbinsx=50, opacity=0.7, marker_color='blue'),
            row=1, col=1
        )

        # Add time-series lines
        for patient_id, group in expanded_df.groupby('PatientID'):
            combined_fig.add_trace(
                Scatter(
                    x=group['SeriesDate'],
                    y=group['HU_Value'],
                    mode='lines',
                    name=f'Patient {patient_id}'
                ),
                row=2, col=1
            )

        # Update layout
        combined_fig.update_layout(
            height=800,
            title_text=f'{HU_selected_metric} Distribution and Time-Series Analysis',
            margin=dict(l=20, r=20, t=50, b=50)
        )

        return combined_fig


    @app.callback(
    Output('patient-dropdown', 'options'),
    Input('ct-scan-data-store', 'data')
    ) 
    def update_patient_dropdown(data):
        if data is None:
            return []

        df = pd.DataFrame(data)
        unique_patient_ids = df['PatientID'].unique()
        return [{'label': str(patient_id), 'value': patient_id} for patient_id in unique_patient_ids]
    
    
    @app.callback(
    Output('z-score-heatmap', 'figure'),
    [Input('ct-scan-data-store', 'data'),
     Input('tissue-dropdown', 'value'),
     Input('hu-threshold-slider', 'value')])
    def update_hu_z_score(data, selected_tissues, hu_threshold):
        if data is None or not selected_tissues:
            return go.Figure()

        # Convert data to DataFrame
        df = pd.DataFrame(data).infer_objects(copy=False)
        
        # Ensure selected_tissues is a list
        selected_tissues = selected_tissues if isinstance(selected_tissues, list) else [selected_tissues]
        
        # Use the selected HU metric from RadioItems
        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues])
        hu_columns = df.columns[df.columns.str.contains(f"{tissue_filter}", regex=True)]
        hu_filtered_df = df[hu_columns].select_dtypes(include='number')

        # Return an empty figure if no HU columns are found
        if hu_filtered_df.empty:
            return go.Figure()

        # Filter data based on HU threshold
        hu_min, hu_max = hu_threshold
        hu_filtered = hu_filtered_df.clip(lower=hu_min, upper=hu_max).dropna()

        # Calculate and clip Z-scores
        z_scores = ((hu_filtered - hu_filtered.mean()) / hu_filtered.std()).clip(-5, 5)
        
        # Map Patient IDs to the filtered data
        patient_ids = df['PatientID'].loc[hu_filtered.index].values

        # Extract region/tissue names for y-axis
        tissue_names = [col.split(';')[1] for col in hu_filtered.columns]
        # Create Z-Score heatmap
        z_score_fig = px.imshow(
            z_scores.T,  # Transpose for better orientation
            labels={'x': 'Patient ID', 'y': 'Tissue Type', 'color': 'Z-Score'},
            x=patient_ids,
            y=tissue_names,
            color_continuous_scale='RdBu',
            aspect="auto"
        )

        # Update layout for improved readability
        z_score_fig.update_layout(
            xaxis_title='Patient ID',
            yaxis_title='Tissue',
            autosize=True,
            height=600,
            width=1400,
            margin=dict(l=20, r=20, t=50, b=50)
        )

        # Hover template for Z-scores
        z_score_fig.update_traces(
            hovertemplate=(
                "<b>Patient ID:</b> %{x}<br>" +
                "<b>Tissue:</b> %{y}<br>" +
                "<b>Z-Score:</b> %{z:.2f}<br>"
            )
        )

        return z_score_fig
        
    # Clustering and Segmentation callback
    @app.callback(
        Output('clustering-plot', 'figure'),
        [
            Input('ct-scan-data-store', 'data'),
            Input('hu-metric-radio', 'value'),
            Input('tissue-dropdown', 'value'),
            Input('slab-dropdown', 'value'),
            Input('hu-threshold-slider', 'value')
        ]
    )
    def update_clustering_plot(data, HU_selected_metric, selected_tissues, selected_slabs, hu_threshold):
        if data is None or not HU_selected_metric or not selected_tissues or not selected_slabs:
            return go.Figure()

        # Load data into DataFrame
        df = pd.DataFrame(data)

        # Ensure required columns exist
        if 'PatientID' not in df.columns:
            print("PatientID column not found.")
            return go.Figure()

        # Filter columns based on dropdown selections
        column_filter = (
            df.columns.str.contains('|'.join(selected_tissues)) &
            df.columns.str.contains('|'.join(selected_slabs)) &
            df.columns.str.contains(HU_selected_metric)
        )
        filtered_columns = df.columns[column_filter]

        if filtered_columns.empty:
            print("No matching columns for the selected filters.")
            return go.Figure()

        # Extract and process the relevant data
        filtered_data = df[filtered_columns].apply(pd.to_numeric, errors='coerce').astype(float)

        # Apply HU threshold range filtering
        hu_min, hu_max = hu_threshold
        filtered_data = filtered_data[(filtered_data >= hu_min) & (filtered_data <= hu_max)]

        # Drop rows with missing values
        filtered_data.dropna(inplace=True)

        if filtered_data.empty:
            print("Filtered data is empty after threshold and NaN removal.")
            return go.Figure()

        # Perform KMeans clustering
        try:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(filtered_data)
            df['Cluster'] = kmeans.labels_
        except Exception as e:
            print(f"Clustering failed: {e}")
            return go.Figure()

        # Prepare data for visualization, including scans and slabs
        cluster_plot_data = pd.DataFrame({
            'PatientID': df['PatientID'].repeat(filtered_columns.shape[0]).reset_index(drop=True),
            'Scan': [col.split(';')[0] for col in filtered_columns] * len(df),  # Extract scan type
            'Slab': [col.split(';')[1] for col in filtered_columns] * len(df),  # Extract slab type
            HU_selected_metric: filtered_data.values.flatten(),
            'Cluster': df['Cluster'].repeat(filtered_columns.shape[0]).reset_index(drop=True)
        })

        # Create the clustering plot
        cluster_fig = px.scatter(
            cluster_plot_data,
            y='Scan',
            x=HU_selected_metric,
            color='Cluster',  
            hover_data={'Slab': True, 'PatientID': True},  
            title='Patient Segmentation & Clustering by Scan and Slab',
            labels={
                'PatientID': 'Patient ID',
                HU_selected_metric: f'{HU_selected_metric} (HU)',
                'Scan': 'Scan Type',
                'Cluster': 'Cluster'
            }
        )

        # Update layout for better visuals
        cluster_fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=50),
            height=800,
            width=1400,
            legend_title_text='Cluster & Scan'
        )

        return cluster_fig

    
    @app.callback(
    Output('vertebrae-count-plot', 'figure'),
    [
        Input('ct-scan-data-store', 'data'),
        Input('slab-dropdown', 'value'),
        Input('tissue-dropdown', 'value'),
        Input('measurement-dropdown', 'value')
    ]
)
    def update_vertebrae_count_plot(data, selected_slabs, selected_tissues, selected_metrics):
        if data is None or not selected_slabs or not selected_tissues or not selected_metrics:
            return go.Figure()

        df = pd.DataFrame(data)

        # List to store presence results
        presence_data = []

        # Iterate over selected slabs to check presence
        for slab in selected_slabs:
            slab_present = False
            for col in df.columns:
                col_parts = col.split(';')  # Split column names into components (adjust if necessary)
                if len(col_parts) < 3:
                    continue
                col_slab, col_tissue, col_metric = col_parts

                # Match selected slab, tissue, and metric
                if (
                    col_slab == slab and 
                    col_tissue in selected_tissues and 
                    col_metric in selected_metrics
                ):
                    if df[col].notnull().any():  # Check if the column has non-null values
                        slab_present = True
                        break

            # Append results for this slab
            presence_data.append({
                'Slab': slab,
                'Presence': 'Present' if slab_present else 'Not Present'
            })

        # Convert presence data to a DataFrame
        presence_df = pd.DataFrame(presence_data)

        # Create a bar plot for slab presence
        vertebrae_fig = px.bar(
            presence_df,
            x='Slab',
            y=[1] * len(presence_df),  
            color='Presence',
            title='Presence of Selected Vertebra Types',
            labels={
                'Slab': 'Vertebra Type',
                'y': 'Presence Indicator'
            },
            color_discrete_map={'Present': 'green', 'Not Present': 'grey'}
        )

        # Update layout
        vertebrae_fig.update_layout(
            xaxis_title='Vertebra Type',
            yaxis_title='',  
            height=600,
            width=1400,
            xaxis={'categoryorder': 'category ascending'},
            hovermode='x',
            showlegend=True
        )

        return vertebrae_fig

    @app.callback(
        Output('tissue-count-plot', 'figure'),
        [Input('ct-scan-data-store', 'data'),
        Input('tissue-dropdown', 'value'),
        Input('slab-dropdown', 'value'),
        Input('measurement-dropdown', 'value')])
    def update_tissue_presence_plot(data, selected_tissues, selected_slabs, selected_metrics):
        if data is None:
            return go.Figure()

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Create filters based on dropdown selections
        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues]) if selected_tissues else ''
        slab_filter = '|'.join([re.escape(slab) for slab in selected_slabs]) if selected_slabs else ''
        metric_filter = '|'.join([re.escape(metric) for metric in selected_metrics]) if selected_metrics else ''

        # Filter columns based on selected tissues, slabs, and metrics
        filtered_columns = df.columns[
            df.columns.str.contains(tissue_filter) &
            df.columns.str.contains(slab_filter) &
            df.columns.str.contains(metric_filter)
        ]

        # Parse the relevant information from column names
        slab_names = [parse_column_name(col)[0] for col in filtered_columns]
        tissue_names = [parse_column_name(col)[1] for col in filtered_columns]

        if filtered_columns.empty:
            return go.Figure()

        # Prepare data for the plot
        result_data = []
        for col in filtered_columns:
            slab, tissue, _ = parse_column_name(col)
            is_present = not df[col].isna().all() 
            tissue_group = classify_tissue(tissue) 
            color = tissue_color_map.get(tissue_group, 'black')  
            patient_ids = df['PatientID'][df[col].notna()].unique()  # Get unique patient IDs with non-null values
            result_data.append({
                'Tissue': tissue,
            'Scan Region': slab,
            'Presence': 'Present' if is_present else 'Not Present',
            'Patient IDs': ', '.join(map(str, patient_ids)) if is_present else '',
            'Color': color,
            'Tissue Group': tissue_group
            })

        # Create DataFrame for plotting
        result_df = pd.DataFrame(result_data)

        # Generate the plot as a heatmap-like display using scatter
        fig = px.scatter(
            result_df,
            x='Tissue',
            y='Scan Region',
            color='Tissue Group',
            symbol='Presence',
            title="Tissue Presence by Scan Region",
            color_discrete_map = tissue_color_map,
            symbol_map={'Present': 'circle', 'Not Present': 'x'}
        )

        fig.update_traces(marker=dict(size=10))  # Adjust marker size for visibility

        fig.update_layout(
            xaxis_title='Tissue Type',
            yaxis_title='Scan Region',
            height=800,
            width=1400
        )

        return fig


    @app.callback(
    [Output('gender-slices-plot', 'figure'), Output('gender-slices-line', 'figure')], 
    Input('ct-scan-data-store', 'data')
)
    def update_gender_slices_plot(data):
        if data is None:
            return go.Figure(), go.Figure()

        # Convert data to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Check if required columns are available
        if 'PatientSex' not in df.columns or 'PatientID' not in df.columns:
            print("Missing required columns: 'PatientSex' or 'PatientID'")
            return go.Figure(), go.Figure()

        # Extract all columns containing slab and slice data (assumes naming convention like 'slab_type;height_mm')
        slab_height_cols = [col for col in df.columns if ';height_mm' in col]

        # If no matching columns are found, return an empty figure
        if not slab_height_cols:
            print("No columns found with ';height_mm' in their names.")
            return go.Figure(), go.Figure()

        # Reshape the DataFrame to have 'slice_type', 'height_mm', and other details in rows
        slice_data = pd.melt(
            df,
            id_vars=['PatientSex', 'PatientID'],  # Non-variable columns
            value_vars=slab_height_cols,         # Columns to unpivot
            var_name='slice_type',               # New column name for the slab type
            value_name='height_mm'               # New column name for the slice height
        )

        # Remove rows with missing or zero height_mm values
        slice_data = slice_data[slice_data['height_mm'] > 0]

        # Sort the data for better visualization
        slice_data = slice_data.sort_values(by=['PatientID', 'slice_type'])

        line_chart = px.scatter(
            slice_data,
            x="slice_type",       # Slab type on the x-axis
            y="height_mm",        # Slice thickness on the y-axis
            color="PatientID",    # Different colors for each Patient ID
            facet_col="PatientSex",  # Separate plots for each gender
            title="Slice Thickness by Slab Type, Patient ID, and Gender",
            labels={
                "height_mm": "Slice Thickness (mm)",
                "slice_type": "Slab Type",
            }
        )

        line_chart.update_layout(
            xaxis_title="Slab Type",
            yaxis_title="Slice Thickness (mm)",
            height=800,
            width=1400,
            legend_title="Patient ID",
            showlegend=True
        )

        # Add histogram data for detailed distribution analysis (optional)
        histogram_data = []
        for patient_id in slice_data['PatientID'].unique():
            patient_data = slice_data[slice_data['PatientID'] == patient_id]
        
            histogram = go.Histogram(
                x=patient_data['slice_type'],  # Slab type
                y=patient_data['height_mm'],  # Slice thickness
                name=f'Patient {patient_id}',  # Label for each patient
                hovertext=patient_data,  # Hover info
                histfunc='avg',  # Average thickness for each slab
                textposition='outside',
                marker=dict(opacity=0.75)
            )
            histogram_data.append(histogram)

        # Create a histogram figure
        histogram_fig = go.Figure(data=histogram_data)

        # Combine both visuals into a final figure (use bar_chart by default)
        return histogram_fig , line_chart
