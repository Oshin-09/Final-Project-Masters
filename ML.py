from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
            dcc.Graph(id='clustering-plot-ml'),  # Updated ID to be unique
            html.H3("Correlation Matrix", style={'text-align': 'center'}),
            dcc.Graph(id='correlation-matrix')  # Added graph for correlation matrix
        ])
    ]
)

# Register callbacks
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

    @app.callback(
        Output('clustering-plot-ml', 'figure'),
        [Input('ct-scan-data-store', 'data'),
         Input('clustering-feature-dropdown', 'value'),
         Input('num-clusters-input', 'value')]
    )
    def perform_clustering(data, selected_features, num_clusters):
        if data is None or not selected_features:
            return go.Figure()

        df = pd.DataFrame(data)
        features = df[selected_features].fillna(0)
        features = StandardScaler().fit_transform(features)

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(reduced_features)

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

    @app.callback(
        Output('correlation-matrix', 'figure'),
        [Input('ct-scan-data-store', 'data'),
         Input('clustering-feature-dropdown', 'value')]
    )
    def plot_correlation_matrix(data, selected_features):
        if data is None or not selected_features:
            return go.Figure()

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Filter selected features and remove non-numeric or NA values
        filtered_df = df[selected_features].select_dtypes(include=[np.number]).dropna()

        if filtered_df.empty:
            return go.Figure()

        # Compute correlation matrix
        correlation_matrix = filtered_df.corr()

        # Plot the heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='Viridis'
            )
        )
        fig.update_layout(
            title="Correlation Matrix (Selected Features)",
            xaxis_title="Features",
            yaxis_title="Features",
            height=800,
            width=1200
        )
        return fig
