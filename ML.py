""" from dash import dcc, html, Input, Output
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
            dcc.Loading(  # Add loading indicators for clustering
                id="loading-clustering",
                type="circle",
                children=dcc.Graph(id='clustering-plot-ml')
            ),
            html.H3("Correlation Matrix", style={'text-align': 'center'}),
            dcc.Loading(  # Add loading indicators for correlation matrix
                id="loading-correlation",
                type="circle",
                children=dcc.Graph(id='correlation-matrix')
            ),
        ])
    ]
)

# Helper function for PCA and clustering
def perform_pca_and_kmeans(data, selected_features, num_clusters):
    # Preprocess features
    features = data[selected_features].fillna(0)
    features = StandardScaler().fit_transform(features)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_features)

    return reduced_features, clusters, pca

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
        # Validate inputs
        if data is None or not selected_features:
            return go.Figure(layout={"title": "Please select at least two features for clustering."})
        if num_clusters is None or num_clusters < 2:
            return go.Figure(layout={"title": "Number of clusters must be at least 2."})

        df = pd.DataFrame(data)

        try:
            reduced_features, clusters, pca = perform_pca_and_kmeans(df, selected_features, num_clusters)
        except Exception as e:
            return go.Figure(layout={"title": f"Error in clustering: {str(e)}"})

        # Create clustering plot
        explained_variance = pca.explained_variance_ratio_[:2].sum()
        clustering_fig = px.scatter(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            color=pd.Series(clusters, name="Cluster").astype(str),
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Cluster'},
            title=f'K-Means Clustering with {num_clusters} Clusters (Explained Variance: {explained_variance:.2f})'
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
            return go.Figure(layout={"title": "Please select at least one feature for the correlation matrix."})

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Filter selected features and remove non-numeric or NA values
        filtered_df = df[selected_features].select_dtypes(include=[np.number]).dropna()

        if filtered_df.empty:
            return go.Figure(layout={"title": "No valid numeric data available for correlation matrix."})

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
        return fig """


from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Layout for the Prediction Tab
ML_layout = dcc.Tab(
    label='Prediction',
    children=[
        html.Div([
            html.H3("Supervised Learning - Prediction"),
            dcc.Dropdown(
                id='prediction-feature-dropdown',
                multi=True,
                placeholder="Select features for prediction",
                style={'margin-bottom': '10px'}
            ),
            dcc.Dropdown(
                id='target-variable-dropdown',
                placeholder="Select target variable",
                style={'margin-bottom': '10px'}
            ),
            html.Button("Train & Predict", id='train-predict-button', n_clicks=0, style={'margin-bottom': '10px'}),
            html.Div(id='prediction-results', style={'margin-top': '20px'}),
            dcc.Graph(id='feature-importance-graph'),
            dcc.Graph(id='actual-vs-predicted-graph'),
            dcc.Graph(id='correlation-matrix-graph-ML')
        ])
    ]
)


def register_callback_ML(app):
    @app.callback(
        [Output('prediction-feature-dropdown', 'options'),
         Output('target-variable-dropdown', 'options')],
        Input('ct-scan-data-store', 'data')
    )
    def update_dropdown_options(data):
        if data is None:
            return [], []
        df = pd.DataFrame(data)
        options = [{'label': col, 'value': col} for col in df.columns]
        return options, options

    @app.callback(
    Output('prediction-results', 'children'),
    [Input('train-predict-button', 'n_clicks'),
     Input('ct-scan-data-store', 'data'),
     Input('prediction-feature-dropdown', 'value'),
     Input('target-variable-dropdown', 'value')]
    )
    def train_and_predict(n_clicks, data, selected_features, target_variable):
        if n_clicks == 0 or data is None or not selected_features or not target_variable:
            return "Please select features and a target variable, then click Train & Predict."

        try:
            df = pd.DataFrame(data)

            # Prepare features and target
            X = df[selected_features].fillna(0)
            y = df[target_variable]

            # Encode categorical features
            if X.select_dtypes(include=['object']).shape[1] > 0:
                X = pd.get_dummies(X)

            # Scale numerical features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Split data into train and test sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a regression model
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            # Predict and evaluate
            predictions = model.predict(X_test)
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            return [
                f"Model trained successfully!",
                f"Mean Squared Error: {mse:.2f}",
                f"R2 Score: {r2:.2f}"
            ]
        except Exception as e:
            return f"Error during training/prediction: {str(e)}"

    @app.callback(
        Output('feature-importance-graph', 'figure'),
        [Input('train-predict-button', 'n_clicks'),
        Input('ct-scan-data-store', 'data'),
        Input('prediction-feature-dropdown', 'value'),
        Input('target-variable-dropdown', 'value')]
    )
    def plot_feature_importance(n_clicks, data, selected_features, target_variable):
        if n_clicks == 0 or data is None or not selected_features or not target_variable:
            return go.Figure(layout={"title": "Train the model to view feature importance."})

        try:
            df = pd.DataFrame(data)

            # Prepare features and target
            X = df[selected_features].fillna(0)
            y = df[target_variable]

            # Train-test split and train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Extract feature importance
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importance})
            fig = px.bar(
                importance_df.sort_values(by="Importance", ascending=False),
                x='Feature', y='Importance',
                title="Feature Importance for Prediction"
            )
            return fig

        except Exception as e:
            return go.Figure(layout={"title": f"Error: {str(e)}"})
    @app.callback(
    Output('actual-vs-predicted-graph', 'figure'),
    [Input('train-predict-button', 'n_clicks'),
     Input('ct-scan-data-store', 'data'),
     Input('prediction-feature-dropdown', 'value'),
     Input('target-variable-dropdown', 'value')]
)
    def plot_actual_vs_predicted(n_clicks, data, selected_features, target_variable):
        if n_clicks == 0 or data is None or not selected_features or not target_variable:
            return go.Figure(layout={"title": "Train the model to view predictions."})

        try:
            df = pd.DataFrame(data)

            # Prepare features and target
            X = df[selected_features]
            y = df[target_variable]

        # Drop rows with NaN values in either features or target
            df_clean = pd.concat([X, y], axis=1).dropna()
            X = df_clean[selected_features]
            y = df_clean[target_variable]

            # Train-test split and train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Create DataFrame for visualization
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

            # Create scatter plot
            fig = px.scatter(
                results_df,
                x='Actual',
                y='Predicted',
                title="Actual vs. Predicted Values",
                labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
            )
            fig.add_shape(
                type='line',
                x0=min(results_df['Actual']),
                y0=min(results_df['Actual']),
                x1=max(results_df['Actual']),
                y1=max(results_df['Actual']),
                line=dict(color='red', dash='dash')
            )
            return fig
        except Exception as e:
            return go.Figure(layout={"title": f"Error: {str(e)}"})

    @app.callback(
    Output('correlation-matrix-graph-ML', 'figure'),
    [Input('train-predict-button', 'n_clicks'),
     Input('ct-scan-data-store', 'data'),
     Input('prediction-feature-dropdown', 'value')]  # Get selected features
)
    def plot_correlation_matrix(n_clicks, data, selected_features):
        if n_clicks == 0 or data is None or not selected_features:
            return go.Figure(layout={"title": "Select features and click Train & Predict to view the correlation matrix."})

        try:
            # Load data into a DataFrame
            df = pd.DataFrame(data)

            # Filter for selected features
            df_filtered = df[selected_features]

            # Drop rows with NA values
            df_clean = df_filtered.dropna()

            # Compute correlation matrix
            correlation_matrix = df_clean.corr()

            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='Viridis',
                    showscale=True
                )
            )
            fig.update_layout(
                title="Correlation Matrix for Selected Features",
                xaxis_title="Features",
                yaxis_title="Features"
            )
            return fig
        except Exception as e:
            return go.Figure(layout={"title": f"Error: {str(e)}"})
