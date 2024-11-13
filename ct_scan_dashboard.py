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
        
        dcc.Tab(label='HU Distribution Patterns', children=[
            # Heatmap display area
            dcc.Graph(id='hu-distribution-plot', style={'margin-top': '30px'})
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

def parse_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    if filename.endswith('.csv'):
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        return None  # Unsupported file format
    
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
        [Input('ct-scan-data-store', 'data'),
        Input('tissue-dropdown', 'value')] )
    def update_time_series_plot(data, selected_tissues):
        if data is None or not selected_tissues:
            return go.Figure()

        df = pd.DataFrame(data)
    
       
        if 'SeriesDate' not in df.columns:
            return go.Figure()
        
        df['SeriesDate'] = df['SeriesDate'].astype(str).str.slice(0, 4) + '/' + \
                   df['SeriesDate'].astype(str).str.slice(4, 6) + '/' + \
                   df['SeriesDate'].astype(str).str.slice(6, 8)

        # Convert 'Timestamp' column to datetime format
        df['SeriesDate'] = pd.to_datetime(df['SeriesDate'], errors='coerce')

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
        filtered_data = df[['SeriesDate'] + list(filtered_columns)]
        time_series_fig = px.line(
            filtered_data, x='SeriesDate', y=filtered_columns[0],
            title='Time-Series Analysis of HU Values'
        )
        time_series_fig.update_layout(
            xaxis_title='Time', yaxis_title='HU Value',
            autosize=False, 
            margin=dict(l=20, r=20, t=50, b=50),  
            height=800,  
            width=1400,   
    )
        return time_series_fig
    
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
        [Input('ct-scan-data-store', 'data'),
        Input('hu-threshold-slider', 'value')])  
    def update_clustering_plot(data, hu_threshold):
        if data is None:
            return go.Figure()

        df = pd.DataFrame(data)
        df = df.infer_objects(copy=False)

        # Ensure PatientID is available as a column
        if 'PatientID' not in df.columns:
            print("PatientID column not found.")
            return go.Figure()
        
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

        # Create the clustering plot using PatientID for x-axis
        cluster_fig = px.scatter(
            df, 
            x='PatientID', 
            y=selected_hu_column, 
            color='Cluster',
            title='Patient Segmentation & Clustering'
        )
        
        cluster_fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=50),
            height=800,
            width=1400
        )

        return cluster_fig


    
    @app.callback(
        Output('vertebrae-count-plot', 'figure'),
        [Input('ct-scan-data-store', 'data'),
        Input('slab-dropdown', 'value'),
        Input('tissue-dropdown', 'value'),
        Input('measurement-dropdown', 'value')]
    )
    def update_vertebrae_count_plot(data, selected_slabs, selected_tissues, selected_metrics):
        if data is None or not selected_slabs or not selected_tissues or not selected_metrics:
            return go.Figure()

        df = pd.DataFrame(data)

        # Initialize a list to store results for each slab
        presence_data = []

        # Check if each selected slab is present with non-null values in the data
        for slab in selected_slabs:
            slab_present = False
            for col in df.columns:
                col_slab, col_tissue, col_metric = parse_column_name(col)
                # Check if this column matches selected slab, tissue, and metric
                if (col_slab == slab and col_tissue in selected_tissues and col_metric in selected_metrics):
                    if df[col].notnull().any():  
                        slab_present = True
                        print(f"Found data for slab {slab} with tissue {col_tissue} and metric {col_metric}")
                        break

            # Record presence of each slab
            presence_data.append({'Slab': slab, 'Presence': 'Present' if slab_present else 'Not Present', 'Value': 1 })

        # Convert presence data to DataFrame for plotting
        presence_df = pd.DataFrame(presence_data)

        print("Presence data:", presence_df)

        # Create a bar plot for presence
        vertebrae_fig = px.bar(
            data_frame=presence_df,
            x='Slab',
            y='Presence',
            title='Presence of Selected Vertebra Types',
            labels={
                'Slab': 'Vertebra Type',
                'Presence': 'Presence Status'
            },
            color='Presence',
            color_discrete_map={'Present': 'green',  'Not Present': 'grey'}
        )

        vertebrae_fig.update_layout(
            xaxis_title='Vertebra Type',
            yaxis_title='Presence Status',
            height=600,
            width=1400,
            xaxis={'categoryorder': 'category ascending'},
            hovermode='x'
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
    Output('gender-slices-plot', 'figure'),
    Input('ct-scan-data-store', 'data'))
    def update_gender_slices_plot(data):
        if data is None:
            return go.Figure()

        df = pd.DataFrame(data)
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

        if 'PatientSex' not in df.columns or 'PatientID' not in df.columns:
            print("Missing 'PatientSex' or 'PatientID' columns.")
            return go.Figure()

        # For each slice, we want to have an individual record
        slice_data = pd.DataFrame()

        # Iterate through each slice column
        for slice_col in num_slices_cols:
            # Create a new dataframe with individual slice info
            slice_info = df[['PatientSex', 'PatientID']].copy()
            slice_info['num_slices'] = df[slice_col]
            
            if len(height_mm_cols) > 0:
                slice_info['height_mm'] = df[height_mm_cols[0]]
            else:
                slice_info['height_mm'] = np.nan

            slice_info['slice_type'] = slice_col
            
            # Filter out rows where 'num_slices' is zero
            slice_info = slice_info[slice_info['num_slices'] > 0]

            # Append this slice's data to the overall slice data
            slice_data = pd.concat([slice_data, slice_info], ignore_index=True)

        # Sort by Patient ID and Gender for better visualization
        slice_data = slice_data.sort_values(by=['PatientID', 'PatientSex'])

        # Assuming 'PatientSex' and other fields are available
        slice_data_fig = px.bar(
            data_frame=slice_data,
            x='slice_type',  
            y='height_mm', 
            color='PatientSex', 
            hover_data=['num_slices', 'PatientID'],
            title='Slice Thickness and Volume per Slab',
            labels={
                'height_mm': 'Slice Thickness (mm)',
                'num_slices': 'Number of Slices',
                'PatientSex': 'PatientSex',
                'PatientID': 'Patient ID',
                'slice_type': 'Slab'
            },
            barmode='group'
        )

        slice_data_fig.update_traces(
            hovertemplate="<b>Slab:</b> %{x}<br>" +
                        "<b>Slice Thickness:</b> %{y} (mm)<br>" +
                        "<b>Number of Slices:</b> %{customdata[0]}<br>" +
                        "<b>Patient ID:</b> %{customdata[1]}<br>"
        )
        slice_data_fig.update_layout(
            xaxis_title='Slab',
            yaxis_title='Slice Thickness',
            autosize=True,
            height=800,
            width=1200
        )

        return slice_data_fig
