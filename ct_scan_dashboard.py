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

tissue_definitions = {
    'Muscle Group': ['ALLSKM', 'LPECMJR', 'RPECMJR', 'LPECMNR', 'RPECMNR', 'LTEMPORALIS', 'RTEMPORALIS', 'LMASSETER', 'RMASSETER'],
    'Adipose Tissue': ['VAT', 'EpAT', 'PaAT', 'ThAT', 'SAT', 'LASAT', 'RASAT'],
    'IMAT': ['ALLIMAT', 'ALLIMAT_NOARMS', 'LUPlGMAT', 'RUPlGMAT', 'LLWLGIMAT', 'RLWLGIMAT', 'LAIMAT', 'RAIMAT'],
    'Organ Metrics': ['LIV', 'SPL', 'LKID', 'RKID', 'HRT', 'PANC', 'BRAIN', 'GB', 'BLADDER', 'PROSTATE', 'AO', 'LLUNG', 'RLUNG', 'TRACH'],
    'Fluid Build-up': ['ASCITES', 'LUNG_EFFUSION'],
    'Cardiovascular': ['AOC', 'CAAC', 'AOC-U-CAAC'],
    'Bones': ['ALLBONE', 'ALLBONE_NOARMS', 'LFMRHEAD', 'RFMRHEAD', 'LFMRNECK', 'RFMRNECK', 'LFMRSHAFT', 'RFMRSHAFT', 'LHPBONE', 'RHPBONE', 'TRBCLR', 'LABONE', 'RABONE'],
}

slab_options = [
    'FULL_SCAN', 'L3mid', 'avg-L3mid[3]', 'T1start-to-L5end', 'L1start-to-L1end', 
    'L4start-to-scanend', 'Head', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4',
    'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'SACRUM', 'LSTV', 'Femur', 'Knee', 'Tibia', 'Feet'
]

measurement_metrics = [
    {'label': 'Cross-sectional Area (pixels)', 'value': 'cross_sectional_area_pixels'},
    {'label': 'Cross-sectional Area (cm²)', 'value': 'cross_sectional_area_cm2'},
    {'label': 'Volume (voxels)', 'value': 'volume_voxels'},
    {'label': 'Volume (cm³)', 'value': 'volume_cm3'},
    {'label': 'HU Mean', 'value': 'HU_mean'},
    {'label': 'HU Standard Deviation', 'value': 'HU_std'},
    {'label': 'HU Min', 'value': 'HU_min'},
    {'label': 'HU Max', 'value': 'HU_max'},
    {'label': 'Image size', 'value': 'img_size_WxLxH'},
    {'label': 'Voxel size', 'value': 'voxel_size_WxLxH'},
    {'label': 'No. of slices', 'value': 'num_slices'}
]

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
        dcc.Tab(label='Z-Score & Correlation', children=[
            html.Div([
                html.H3("Z-Score Map", style={'text-align': 'center'}),
                dcc.Graph(id='z-score-heatmap'),
                html.H3("Correlation Matrix", style={'text-align': 'center'}),
                dcc.Graph(id='correlation-matrix')
            ])
        ]),
        dcc.Tab(label='3D Visualization', children=[
            html.Div([
                html.H3("3D Visualization of HU Values", style={'text-align': 'center'}),
                dcc.Graph(id='3d-plot')
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
        dcc.Tab(label='Gender Distribution', children=[
            html.Div([
               html.H3("Gender Distribution", style={'text-align': 'center'}),
               dcc.Graph(id='gender-distribution-plot')
               ])
        ]),
         dcc.Tab(label='Slices per Patient', children=[
            html.Div([
               html.H3("Slices per Patient", style={'text-align': 'center'}),
               dcc.Graph(id='slices-per-patient-plot')
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
        
        if filtered_data.empty:
            return go.Figure(), go.Figure()
        

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=filtered_data.iloc[:, 0],
            x=filtered_columns.values,
            y=filtered_data.index,
            colorscale='Jet',
            colorbar=dict(title="HU Value"),
        ))
        heatmap_fig.update_layout(autosize=False, margin=dict(l=20, r=20, t=50, b=50), height=600,width=1400)

        hist_fig = px.histogram(filtered_data, x=filtered_data.columns[0], nbins=50, title='HU Distribution',
                                marginal='rug', opacity=0.7, color_discrete_sequence=['blue'])
        hist_fig.update_layout(xaxis_title='HU Value', yaxis_title='Frequency' , autosize=False, margin=dict(l=20, r=20, t=50, b=50), height=600,width=1400 
    )

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


    # Z-Score Heatmap and Correlation Matrix callback
    @app.callback(
    [Output('z-score-heatmap', 'figure'), Output('correlation-matrix', 'figure')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('tissue-dropdown', 'value')])
    def update_z_score_and_correlation(contents, filename, selected_tissues):
        if contents is None or not selected_tissues:
            return go.Figure(), go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure(), go.Figure()
        df.interpolate(method='linear', inplace=True)

        # Ensure selected_tissues is a list
        if not isinstance(selected_tissues, list):
            selected_tissues = [selected_tissues]

        # Filter columns that match the selected tissues
        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues])
        filtered_columns = df.columns[df.columns.str.contains(tissue_filter, regex=True)]

        if len(filtered_columns) == 0:
            return go.Figure(), go.Figure()

        # Z-Score calculation
        filtered_data = df[filtered_columns]

        z_scores = (filtered_data - filtered_data.mean()) / filtered_data.std()


        z_threshold = 0
        z_scores_highlighted = z_scores.applymap(lambda x: x if abs(x) > z_threshold else None)

        z_score_fig = go.Figure(data=go.Heatmap(
            z=z_scores_highlighted.values,  # Highlighted Z-scores
            x=filtered_columns.values,
            y=filtered_data.index,
            colorscale='RdBu',
            colorbar=dict(title="Z-Score"),
            zmin=-z_threshold, zmax=z_threshold  # Limit color scale to threshold
        ))
        z_score_fig.update_layout(autosize=False, margin=dict(l=20, r=20, t=50, b=50), height=600, width=1400)


        def calculate_correlation_and_pvalues(df):
            df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
            df = df.dropna() 
            df_corr = df.corr(method='pearson')
            p_values = pd.DataFrame(index=df.columns, columns=df.columns)
                
            for row in df.columns:
                for col in df.columns:
                        corr, p_val = stats.pearsonr(df[row], df[col])
                        p_values.at[row, col] = p_val
                return df_corr, p_values

        # Get correlation matrix and p-values
        corr_matrix, p_value_matrix = calculate_correlation_and_pvalues(filtered_data)

        # Create correlation matrix heatmap
        corr_matrix_fig = px.imshow(corr_matrix, title='Correlation Matrix Grouped by Tissue Type',
                                    color_continuous_scale='RdBu', zmin=-1, zmax=1, aspect="auto")
        corr_matrix_fig.update_traces(hovertemplate='Correlation: %{z:.2f}')
        corr_matrix_fig.update_layout(xaxis_title='Features', yaxis_title='Features',
                                    autosize=False, margin=dict(l=20, r=20, t=50, b=50), height=600, width=1400)

        return z_score_fig, corr_matrix_fig


    @app.callback(
    Output('3d-plot', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('tissue-dropdown', 'value')])
    
    def update_3d_visualization(contents, filename, selected_tissues):
        if contents is None or not selected_tissues:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()
        df.interpolate(method='linear', inplace=True)

        # Ensure selected_tissues is a list
        if not isinstance(selected_tissues, list):
            selected_tissues = [selected_tissues]

        # Filter columns that match the selected tissues
        tissue_filter = '|'.join([re.escape(tissue) for tissue in selected_tissues])
        filtered_columns = df.columns[df.columns.str.contains(tissue_filter, regex=True)]

        # Ensure there are at least three columns for 3D plotting
        if len(filtered_columns) < 3:
            return go.Figure()

        # Create 3D scatter plot with the first three filtered columns
        filtered_data = df[filtered_columns[:3]]
        scatter_3d_fig = px.scatter_3d(
            filtered_data, x=filtered_columns[0], y=filtered_columns[1], z=filtered_columns[2],
            color=filtered_columns[0], title='3D Visualization of HU Values'
        )
        scatter_3d_fig.update_layout(
    autosize=False,  # Disable autosizing to manually control
    margin=dict(l=20, r=20, t=50, b=50),  # Tighter margins
    height=600,  # Adjust height as needed
    width=1400,   # Adjust width as needed
    )

        return scatter_3d_fig
    
    # Clustering and Segmentation callback
    @app.callback(
        Output('clustering-plot', 'figure'),
        [Input('upload-data', 'contents'),
        Input('upload-data', 'filename')])
    def update_clustering_plot(contents, filename):
        if contents is None:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()
        df.interpolate(method='linear', inplace=True)

        # Automatically detect the column containing 'HU' in the name
        hu_columns = [col for col in df.columns if 'HU' in col]

        if len(hu_columns) == 0:
            return go.Figure()  # Return an empty figure if no HU columns found

        # Use the first HU-related column for clustering
        selected_hu_column = hu_columns[0]

        # Perform KMeans clustering on the HU data
        kmeans = KMeans(n_clusters=3).fit(df[[selected_hu_column]])
        df['Cluster'] = kmeans.labels_

        # Create the clustering plot
        cluster_fig = px.scatter(df, x=df.index, y=selected_hu_column, color='Cluster',
                                title='Patient Segmentation & Clustering')
        cluster_fig.update_layout(
    autosize=False,  # Disable autosizing to manually control
    margin=dict(l=20, r=20, t=50, b=50),  # Tighter margins
    height=600,  # Adjust height as needed
    width=1400,   # Adjust width as needed
    )

        return cluster_fig


    @app.callback(
        Output('gender-distribution-plot', 'figure'),
        [Input('upload-data', 'contents'),
        Input('upload-data', 'filename')]
    )
    def update_gender_distribution_plot(contents, filename):
        if contents is None:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()

        # Group by PatientSex and count the number of unique PatientIDs
        gender_counts = df.groupby('Gender')['PatientID'].nunique()

        # Create a pie chart or bar chart for gender distribution
        gender_fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index,
                            title='Gender Distribution of Patients')
        return gender_fig

    @app.callback(
    Output('vertebrae-count-plot', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('slab-dropdown', 'value')] )
    def update_vertebrae_count_plot(contents, filename, selected_scans):
        if contents is None:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()

        # Filter based on selected scan types (e.g., L3mid, L5end, etc.)
        scan_filter = '|'.join([re.escape(scan) for scan in selected_scans]) if selected_scans else ''
        filtered_columns = df.columns[df.columns.str.contains(scan_filter)]

        slab_names = [parse_column_name(col)[0] for col in filtered_columns]


        if filtered_columns.empty:
            return go.Figure()

        # Count the occurrences of each scan type
        vertebrae_count = df[filtered_columns].count()

        # Create a bar chart for the vertebrae count
        vertebrae_fig = px.bar(x=slab_names, y=vertebrae_count.values,
                            title='Vertebrae Count Based on Scan Types',
                            labels={'x': 'Scan Type', 'y': 'Count'})
        vertebrae_fig.update_layout( xaxis_title='Scan Type', yaxis_title='Count',
    autosize=False,  # Disable autosizing to manually control
    margin=dict(l=20, r=20, t=50, b=50),  # Tighter margins
    height=600,  # Adjust height as needed
    width=1400,   # Adjust width as needed
    )

        return vertebrae_fig

    @app.callback(
    Output('slices-per-patient-plot', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('slab-dropdown', 'value'),
     Input('measurement-dropdown', 'value')])
    
    def update_slices_per_patient_plot(contents, filename, selected_scans, selected_metrics):
        if contents is None:
            return go.Figure()

        df = parse_file(contents, filename)
        if df is None:
            return go.Figure()

        # Filter columns for 'num_slices'
        scan_filter = '|'.join([re.escape(scan) for scan in selected_scans]) if selected_scans else ''
        metric_filter = '|'.join([re.escape(metric) for metric in selected_metrics]) if selected_metrics else ''

        # Check if 'num_slices' exists in any of the columns
        filtered_columns = df.columns[
            df.columns.str.contains(scan_filter) &
            df.columns.str.contains(metric_filter)
        ]

        filtered_data = df[filtered_columns]

        if 'PatientID' in df.columns:
            # Check if there's a column containing 'num_slices'
            num_slices_column = [col for col in filtered_columns if 'num_slices' in col]
            if num_slices_column:
                # Count unique slices per patient
                slices_per_patient = df.groupby('PatientID')[num_slices_column[0]].nunique()
                
                if not slices_per_patient.empty:
                    # Create a bar chart showing the number of slices per patient
                    fig = px.bar(
                        slices_per_patient.reset_index(),
                        x='PatientID',
                        y=num_slices_column[0],
                        title='Number of Unique Slices Per Patient',
                        labels={'PatientID': 'Patient ID', num_slices_column[0]: 'Number of Slices'}
                    )
                    fig.update_layout(xaxis_title='Patient ID', yaxis_title='Number of Slices',
    autosize=False,  # Disable autosizing to manually control
    
    height=600,  # Adjust height as needed
    width=1400,   # Adjust width as needed
)
                    return fig

        return go.Figure()
