import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from skellyalign.models.recording_config import RecordingConfig
from skellyalign.utilities.data_handlers import DataLoader, DataProcessor
from skellyalign.align_things import get_best_transformation_matrix_ransac, apply_transformation
import plotly.express as px
import plotly.graph_objs as go

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SkellyAlign"),

    html.Label("FreeMoCap Path:"),
    dcc.Input(id='freemocap-path', type='text', value='', style={'width': '100%'}),
    html.Br(),

    html.Label("Qualisys Path:"),
    dcc.Input(id='qualisys-path', type='text', value='', style={'width': '100%'}),
    html.Br(),

    html.Label("Markers for Alignment (comma separated):"),
    dcc.Input(id='markers-for-alignment', type='text', value='', style={'width': '100%'}),
    html.Br(),

    html.Label("FreeMoCap Markers (comma separated):"),
    dcc.Input(id='freemocap-markers', type='text', value='', style={'width': '100%'}),
    html.Br(),

    html.Label("Qualisys Markers (comma separated):"),
    dcc.Input(id='qualisys-markers', type='text', value='', style={'width': '100%'}),
    html.Br(),

    html.Label("Frames to Sample:"),
    dcc.Input(id='frames-to-sample', type='number', value=20, style={'width': '100%'}),
    html.Br(),

    html.Label("Max Iterations:"),
    dcc.Input(id='max-iterations', type='number', value=20, style={'width': '100%'}),
    html.Br(),

    html.Label("Inlier Threshold:"),
    dcc.Input(id='inlier-threshold', type='number', value=50, style={'width': '100%'}),
    html.Br(),

    html.Button('Run Alignment', id='run-alignment', n_clicks=0),
    html.Div(id='output-message'),
    dcc.Graph(id='alignment-plot')
])

@app.callback(
    [Output('output-message', 'children'),
     Output('alignment-plot', 'figure')],
    [Input('run-alignment', 'n_clicks')],
    [State('freemocap-path', 'value'),
     State('qualisys-path', 'value'),
     State('markers-for-alignment', 'value'),
     State('freemocap-markers', 'value'),
     State('qualisys-markers', 'value'),
     State('frames-to-sample', 'value'),
     State('max-iterations', 'value'),
     State('inlier-threshold', 'value')]
)
def run_alignment(n_clicks, freemocap_path, qualisys_path, markers_for_alignment, freemocap_markers, qualisys_markers, frames_to_sample, max_iterations, inlier_threshold):
    if n_clicks == 0:
        return "", go.Figure()

    try:
        # Create recording config
        recording_config = RecordingConfig(
            path_to_recording="",
            path_to_freemocap_output_data=freemocap_path,
            path_to_qualisys_output_data=qualisys_path,
            freemocap_markers=freemocap_markers.split(','),
            qualisys_markers=qualisys_markers.split(','),
            markers_for_alignment=markers_for_alignment.split(','),
            frames_to_sample=frames_to_sample,
            max_iterations=max_iterations,
            inlier_threshold=inlier_threshold
        )

        # Load and process FreeMoCap data
        freemocap_dataloader = DataLoader(path=recording_config.path_to_freemocap_output_data)
        freemocap_data_processor = DataProcessor(data=freemocap_dataloader.data, marker_list=recording_config.freemocap_markers, markers_for_alignment=recording_config.markers_for_alignment)
        freemocap_extracted = freemocap_data_processor.extracted_data_3d

        # Load and process Qualisys data
        qualisys_dataloader = DataLoader(path=recording_config.path_to_qualisys_output_data)
        qualisys_data_processor = DataProcessor(data=qualisys_dataloader.data, marker_list=recording_config.qualisys_markers, markers_for_alignment=recording_config.markers_for_alignment)
        qualisys_extracted = qualisys_data_processor.extracted_data_3d

        # Run RANSAC alignment
        best_transformation_matrix = get_best_transformation_matrix_ransac(
            freemocap_data=freemocap_extracted,
            qualisys_data=qualisys_extracted,
            frames_to_sample=recording_config.frames_to_sample,
            max_iterations=recording_config.max_iterations,
            inlier_threshold=recording_config.inlier_threshold
        )
        aligned_freemocap_data = apply_transformation(best_transformation_matrix, freemocap_dataloader.data)

        # Create a scatter plot of the aligned data
        fig = go.Figure()

        # Add FreeMoCap data
        for i in range(aligned_freemocap_data.shape[1]):
            fig.add_trace(go.Scatter3d(
                x=aligned_freemocap_data[:, i, 0],
                y=aligned_freemocap_data[:, i, 1],
                z=aligned_freemocap_data[:, i, 2],
                mode='markers',
                name=f'FreeMoCap Marker {i}'
            ))

        # Add Qualisys data
        for i in range(qualisys_dataloader.data.shape[1]):
            fig.add_trace(go.Scatter3d(
                x=qualisys_dataloader.data[:, i, 0],
                y=qualisys_dataloader.data[:, i, 1],
                z=qualisys_dataloader.data[:, i, 2],
                mode='markers',
                name=f'Qualisys Marker {i}'
            ))

        fig.update_layout(title='Aligned FreeMoCap and Qualisys Data', scene=dict(aspectmode='data'))

        return "Alignment successful", fig

    except Exception as e:
        return f"Error: {e}", go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
