import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import numpy as np

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
target_names = iris.target_names

# Create Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("🌼 Iris Species Classifier", style={'textAlign': 'center'}),

    html.Label("🌿 Choose input features:"),
    dcc.Dropdown(
        id='feature-selector',
        options=[{'label': col, 'value': col} for col in iris.feature_names],
        value=['sepal length (cm)', 'petal length (cm)'],
        multi=True
    ),

    dcc.Graph(id='classification-graph'),
    html.Div(id='accuracy', style={'fontSize': 20, 'marginTop': '20px'}),

    html.H3("🔍 Confusion Matrix"),
    dcc.Graph(id='confusion-matrix'),

    html.H3("📊 Classification Report (F1, Precision, Recall)"),
    html.Pre(id='report', style={'whiteSpace': 'pre-wrap', 'fontSize': 16})
])

@app.callback(
    Output('classification-graph', 'figure'),
    Output('accuracy', 'children'),
    Output('confusion-matrix', 'figure'),
    Output('report', 'children'),
    Input('feature-selector', 'value')
)
def update_classification(features):
    if len(features) < 1:
        return {}, "⚠️ Please select at least one feature.", {}, ""

    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = RandomForestClassifier(n_estimators=50, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    labels = list(target_names)

    # Main classification chart
    if len(features) == 2:
        fig = px.scatter(
            X_test,
            x=features[0],
            y=features[1],
            color=[target_names[i] for i in y_pred],
            title="🌸 Predicted Classes (Test Set)"
        )
    else:
        fig = px.histogram(x=[target_names[i] for i in y_pred], title="🌸 Predicted Class Distribution")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )
    cm_fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )

    # Classification Report (per-class F1 etc.)
    report_text = classification_report(y_test, y_pred, target_names=labels)

    return fig, f"✅ Accuracy: {acc:.2%}", cm_fig, report_text

if __name__ == '__main__':
    app.run()
