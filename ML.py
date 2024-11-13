# ML.py

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Define the layout for the ML Tab
ML_layout = dcc.Tab(
    label='Machine Learning - Clustering',
    children=[
        html.Div([
            html.H3("K-Means Clustering of CT Scan Data", style={'text-align': 'center'}),
            dcc.Dropdown(
                id='clustering-feature-dropdown',
                multi=True,
                placeholder="Select features for clustering"
            ),
            dcc.Input(
                id='num-clusters-input',
                type='number',
                value=3,
                min=2,
                max=10,
                step=1,
                placeholder="Number of Clusters",
                style={'margin-top': '10px'}
            ),
            dcc.Graph(id='clustering-plot-ml')  # Updated ID to be unique
        ])
    ]
)




def register_callback_ML(app):

    @app.callback(
    Output('clustering-feature-dropdown', 'options'),
    Input('ct-scan-data-store', 'data')
)
    def update_feature_dropdown_options(data):
        if data is None:
            return []

        df = pd.DataFrame(data)
        return [{'label': col, 'value': col} for col in df.columns]
    # Callback for Clustering
    @app.callback(
    Output('clustering-plot-ml', 'figure'),
    [Input('ct-scan-data-store', 'data'),
    Input('clustering-feature-dropdown', 'value'),
    Input('num-clusters-input', 'value')]
)
    def perform_clustering(data, selected_features, num_clusters):
    # Return an empty figure if there's no data or no selected features
        if data is None:
            print("No data available in ct-scan-data-store.")
            return go.Figure()

        if not selected_features:
            print("No features selected.")
            return go.Figure()

    # Proceed with clustering
        df = pd.DataFrame(data)
        features = df[selected_features].fillna(0)
        # Check if all selected features are present in the data
        missing_features = [feature for feature in selected_features if feature not in df.columns]
        if missing_features:
            print(f"Missing features in data: {missing_features}")
            return go.Figure()

        # Proceed with clustering
        features = df[selected_features].fillna(0)  # Optionally handle NaN values
        features = StandardScaler().fit_transform(features)

        # Apply PCA (if desired) for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)

        # K-Means Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(reduced_features)

        # Plotting
        clustering_fig = px.scatter(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            color=df['Cluster'].astype(str),
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Cluster'},
            title=f'K-Means Clustering with {num_clusters} Clusters'
        )

        clustering_fig.update_layout(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            height=800,
            width=1200,
            coloraxis=dict(colorbar=dict(title="Cluster"))
        )

        return clustering_fig
