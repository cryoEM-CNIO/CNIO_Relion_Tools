#!/usr/bin/env python3

"""
Follow relion live
Nayim Gonzalez-Rodriguez, Emma Arean-Ulloa & Rafael Fernandez-Leiro 2022
"""

"""
Activate conda environment before running
relion_live.py should be run from relion's project directory
if your folder has lots of jobs might take a while to load the first time!
"""

### Setup
import os
import pandas as pd
import starfile
import dash
from dash import html
from dash import dcc, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import socket
import argparse
import json
from PIL import Image

# Parsing port number, host and debug mode
parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", help = "choose port to run the webapp")
parser.add_argument("--host", "-host", help = "choose host to run the webapp")
parser.add_argument("--debug", "-d", help = "launch app in debug mode")
args, unknown = parser.parse_known_args()

# Set localhost and 8050 as host and port by default
if not args.port: port_number = 8050
else: port_number = args.port
if not args.host: hostname = socket.gethostname()
else: hostname = args.host
if not args.debug: debug_mode = False
else: debug_mode = args.debug

## Function Definitions

# No-data graph

def empty_graph(text=None, fontsize=20):
    fig = go.Figure()
    if text:
        fig.add_annotation(text=text, showarrow=False, font = dict(size = fontsize))
    fig.update_layout(template="simple_white", showlegend=False, height=220)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

# Zoom images

def image_to_figure(src, width, height, scale_factor):
        # Create figure
    fig = go.Figure()

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, width * scale_factor],
            y=[0, height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, height * scale_factor],
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=width * scale_factor,
            y=height * scale_factor,
            sizey=height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=src)
    )

    # Configure other layout
    fig.update_layout(
        width=width * scale_factor,
        height=height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig

# Scatter plots
def plot_scatter(df, range_y, range_x, title_text, title_y, coloring_selected_mics, coloring_column, filter_upper=999, filter_bottom=999):
    plot = px.scatter(
        data_frame = df,
        range_y = (range_y),
        range_x = (range_x),
        y = title_y,
        color = coloring_column,
        color_discrete_sequence = [coloring_selected_mics,'grey'],
        color_discrete_map={True: coloring_selected_mics, False: 'grey'},
        marginal_y="histogram",
        hover_data=[title_y],
        orientation = 'h',
        render_mode = 'webgl',
        template = 'plotly_white',
        height = 220, #maybe this can be done in a different way?
    )
    plot.update_layout(showlegend=False, xaxis={"showgrid": False}, yaxis={"showgrid": True, "title_text":title_text})
    plot.add_hrect(y0=filter_upper, y1=999, line_width=0, fillcolor="grey", opacity=0.2)
    plot.add_hrect(y0=-999, y1=filter_bottom, line_width=0, fillcolor="grey", opacity=0.2) 
    return plot


# Get dropdown with available jobs 
def get_jobs_list(pipeline_starfile, process):
    processes_df = starfile.read(pipeline_starfile)['pipeline_processes']
    lista = list(processes_df.query(f"rlnPipeLineProcessTypeLabel == 'relion.{process}'")["rlnPipeLineProcessName"])
    jobs = []
    for i in lista:
        jobs.append(i.split("/")[1])
    return jobs

# Merge the filters of different parameters
def merge_filters(filter_df_original, filter_df_added, column_original, column_added):
    if filter_df_added: # If there's already some filter for this parameter...
        ReadIn = pd.DataFrame.from_dict(json.loads(filter_df_added)).reset_index(drop=True)
        filter_df_original = filter_df_original.reset_index(drop=True)
        if len(ReadIn[column_added]) == len(filter_df_original[column_original]):# To be able to merge filtering columns they have to be the same length
            filter_df_original[column_original] = filter_df_original[column_original].mul(ReadIn[column_added])
        else:
            diff_of_rows = len(ReadIn[column_added]) - len(filter_df_original[column_original])
            column_added_filled = filter_df_original[column_original]
            for i in range(diff_of_rows): # Fill up short column with False elements
                column_added_filled += False
            filter_df_original[column_original] = filter_df_original[column_original].mul(column_added_filled)
        filter_df_original.index += 1
    return filter_df_original

## Style
# Header
header_style = {'width': '29%', 'display': 'inline-block', 'vertical-align': 'middle'}
title1_style = {"margin-left": "15px", "margin-top": "15px", "margin-bottom": "0em", "color": "Black", "font-family" : "Helvetica", "font-size":"2.5em"}
title2_style = {"margin-left": "15px", "margin-top": "0em", "color": "Black", "font-family" : "Helvetica"}
progress_style = {'width': '70%', 'display': 'inline-block', 'vertical-align': 'top' , 'marginBottom':0, 'marginTop':-25}

# Tabs
general_text_style_soft = {"margin-left": "15px", "color": "Grey", "font-family" : "Helvetica", 'display': 'inline-block'}
general_text_style = {"margin-left": "15px", "color": "Black", "font-family" : "Helvetica", 'display': 'inline-block'}
dropdown_group_style = {'display': 'inline-block', 'verticalAlign': 'center', 'padding': '20px'}
dropdown_style = {'font-size':'14px',"margin-left": "5px", "margin-right": "10px", 'margin-top': "0px", 'display': 'inline-block', "width": "80px",'border-radius': '20px','vertical-align': 'middle'}
slider_style_h = {'display':'grid', 'grid-template-columns': '10% 25%', 'align-items': 'center'}
TwoGraph_style = {'display':'inline-block', 'width': '50%', 'vertical-align': 'top'}
OneGraph_style = {'display':'inline-block', 'width': '100%', 'vertical-align': 'top', "margin-left": "50px", "margin-right": "50px"}
slider_style_v = {'display':'grid', 'grid-template-columns': '5% 90%', 'align-items': 'top',"margin-left": "2%"}
slider_style_m = {"display": "grid", "grid-template-columns": "85% 5%", 'align-items': 'center'}
tabs_style = {'height': '3em', 'width': '100%', 'display': 'inline', 'vertical-align': 'bottom', 'borderBottom':'3px #000000'}
tab_style = {'padding':'0.5em', "font-family" : "Helvetica", 'background-color':'white'}
tab_selected_style = {'padding':'0.5em', 'borderTop': '3px solid #000000', "font-family" : "Helvetica", 'font-weight':'bold'}
box1_syle = {"margin-left": "15px", "color": "Black", "font-family" : "Helvetica", 'display': 'inline-block','width': '6%', 'height': '5px', 'border': '2px solid grey', 'border-radius': '10px', 'padding': '8px'}
box2_syle = {"margin-left": "15px", "color": "Black", "font-family" : "Helvetica", 'display': 'inline-block','width': '50%', 'height': '10%', 'border': '2px solid grey', 'border-radius': '10px', 'padding': '10px'}

# Buttons

bt_style = {"align-items": "center", "background-color": "#F2F3F4", "border": "2px solid #000",
            "box-sizing": "border-box", "color": "#000", "cursor": "pointer", "display": "inline-flex",
            "font-family": "Helvetica", "margin-bottom":"3px", 'padding':'0.3em 1.2em', 'margin':'0em 0.3em 0.3em 1em',
            "font-size": "0.9em", 'font-weight':'500', 'padding':'0.3em 1.2em', 'border-radius':'2em', 'text-align':'center',
            'transition':'all 0.2s'}


# Colors definitions
color_ctf1 = '#6FC381'
color_ctf2 = '#9AD5A7'
color_ice1 = 'cornflowerblue'
color_ice2 = 'lightblue'
color_res1 = 'salmon'
color_motion1 = '#F6AE2D'
color_motion2 = '#F9CB76'


### Project directory

relion_wd = os.getcwd()

print('starting up relion_live dashboard in '+str(relion_wd)+ ' ...')

### Initialising dash APP

assets_path = relion_wd # this reads the whole folder (!!) so takes long if it is a big project 

app = dash.Dash(
    __name__,
    assets_folder=assets_path,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "RELION Live Dashboard"
server = app.server

### APP Layout

app.layout = html.Div([

    ## Title
    html.Div([
        html.Div([
            html.H1("RELION", style=title1_style),
            html.H2("Live Dashboard", style=title2_style),
            html.Div([dcc.Store(id='filter-motion-data-store')]),
            html.Div([dcc.Store(id='filter-ctf-data-store')]),
            html.Div([dcc.Store(id='filter-ice-data-store')]),
            html.Div([dcc.Store(id='filtered-df-export-store')]),
            html.P(id='placeholder'),
            html.H5("Relion working directory: "+relion_wd, style=general_text_style),
            ], style=header_style),
        html.Div([
            dcc.Graph(id='progress_graph', figure={}),
            ], style=progress_style),
    ]),        
    #Live updates
    dcc.Interval(
        id='interval-component',
        interval=10*1000, # in milliseconds
        n_intervals=0
        ),

        ## Import / MC / CtfFind / Ice job folder inputs

        html.Div([
            html.Div([
                html.H5("Import:", style=general_text_style),
                dcc.Dropdown(id= 'import_name', options=[],value=None,style=dropdown_style),
                html.H5("MotionCorr:", style=general_text_style),
                dcc.Dropdown(id='motion_name', options=[],value=None,style=dropdown_style),
                html.H5("CtfFind:", style=general_text_style),
                dcc.Dropdown(id='ctf_name', options=[],value=None,style=dropdown_style),
                html.H5("Ice Thickness:", style=general_text_style),
                dcc.Dropdown(id='ice_name', options=[],value=None,style=dropdown_style),
            ],style=dropdown_group_style),
            
            html.Div(style={'flex-grow': '1'}),  # Empty div with flex-grow property

            ## Filtering and exporting
            html.Div([
                html.H5("You have selected", style=general_text_style), 
                html.H5(id='filtered-df-export-len', style=general_text_style),
                html.H5("micrographs", style=general_text_style),
                html.Button('Export selected micrographs', id='export-filtered-data-button', style=bt_style),
                ]),
        ],style={'display': 'flex', 'flex-wrap': 'wrap'}),

        ## Graphs

        ### x-slider
        html.Div([
                html.H5("General x-axis range:", style=general_text_style),
                dcc.RangeSlider(id='xrange', min=1, max=300, step=1, value=[], marks=None, 
                                tooltip={"placement": "top", "always_visible": True}),
        ],style=slider_style_h),

        ### block-1
        html.Div([
            ### motion
            html.Div([
                html.Div([
                        html.H5("Total motion: ", style=general_text_style),
                        html.H5("Filter values from", style=general_text_style_soft),
                        dcc.Input(id='motion_lower_limit_filter_input', type='number', value=0,
                                debounce=True, style=box1_syle),
                        html.H5("to", style=general_text_style_soft),
                        dcc.Input(id='motion_upper_limit_filter_input', type='number', value=999,
                                debounce=True, style=box1_syle),
                ]),                    
                html.Div([
                    dcc.RangeSlider(id='totalmotion_minmax', min=0, max=300, step=1, value=[0,300], marks=None,
                                    tooltip={"placement": "left", "always_visible": True},vertical=True, verticalHeight=100),
                    dcc.Store(id='filtered_motion_df'),
                    dcc.Graph(id='totalmotion_graph', figure={}),
                    dcc.Store(id='motion_len'),
                ],style=slider_style_v),                    
            ],style=TwoGraph_style),

            ### ICE
            html.Div([
                html.Div([
                        html.H5("Ice thickness", style=general_text_style),
                        html.H5("Filter values from", style=general_text_style_soft),
                        dcc.Input(id='ice_lower_limit_filter_input', type='number', value=0,
                                debounce=True, style=box1_syle),
                        html.H5("to", style=general_text_style_soft),
                        dcc.Input(id='ice_upper_limit_filter_input', type='number', value=999,
                                debounce=True, style=box1_syle),
                ]),

                html.Div([
                        dcc.RangeSlider(id='ice_minmax', min=0, max=25, step=1, value=[0,20],  marks=None,
                                        tooltip={"placement": "left", "always_visible": True},vertical=True, verticalHeight=100),
                        dcc.Graph(id='ice_graph', figure={}),
                        dcc.Store(id='ice_len'),
                ],style=slider_style_v),                    
            ],style=TwoGraph_style),
        ]),

        ### block-2
        html.Div([
        ### defocus
            html.Div([
                html.Div([
                        html.H5("Defocus", style=general_text_style),
                        html.H5("Filter values from", style=general_text_style_soft),
                        dcc.Input(id='defocus_lower_limit_filter_input', type='number', value=0,
                                debounce=True, style=box1_syle),
                        html.H5("to", style=general_text_style_soft),
                        dcc.Input(id='defocus_upper_limit_filter_input', type='number', value=999,
                                debounce=True, style=box1_syle),
                ]),

                html.Div([
                        dcc.RangeSlider(id='defocus_minmax', min=0, max=6, step=0.2, value=[0,6], marks=None,
                                        tooltip={"placement": "left", "always_visible": True},vertical=True, verticalHeight=100),
                        dcc.Graph(id='defocus_graph', figure={}),
                ],style=slider_style_v),                    
            ],style=TwoGraph_style),

        ### Astigmatism
            html.Div([
                html.Div([
                        html.H5("Astigmatism", style=general_text_style),
                        html.H5("Filter values from", style=general_text_style_soft),
                        dcc.Input(id='astigmatism_lower_limit_filter_input', type='number', value=0,
                                debounce=True, style=box1_syle),
                        html.H5("to", style=general_text_style_soft),
                        dcc.Input(id='astigmatism_upper_limit_filter_input', type='number', value=999,
                                debounce=True, style=box1_syle),
                ]),

                html.Div([
                        dcc.RangeSlider(id='astigmatism_minmax', min=-0.05, max=0.5, step=0.01, value=[-0.05,0.15],  marks=None,
                                        tooltip={"placement": "left", "always_visible": True},vertical=True, verticalHeight=100),
                        dcc.Graph(id='astigmatism_graph', figure={}),
                ],style=slider_style_v),                    
            ],style=TwoGraph_style),
        ]),

        ### block-3
        html.Div([
        ### MaxRes
            html.Div([
                html.Div([
                        html.H5("Max Resolution", style=general_text_style),
                        html.H5("Filter values from", style=general_text_style_soft),
                        dcc.Input(id='maxres_lower_limit_filter_input', type='number', value=0,
                                debounce=True, style=box1_syle),
                        html.H5("to", style=general_text_style_soft),
                        dcc.Input(id='maxres_upper_limit_filter_input', type='number', value=999,
                                debounce=True, style=box1_syle),
                ]),

                html.Div([
                        dcc.RangeSlider(id='maxres_minmax', min=0, max=60, step=0.5, value=[0,10],  marks=None,
                                        tooltip={"placement": "left", "always_visible": True},vertical=True, verticalHeight=100),
                        dcc.Graph(id='maxres_graph', figure={}),
                ],style=slider_style_v),                    
            ],style=TwoGraph_style),

            ### FOM
            html.Div([
                html.Div([
                        html.H5("CTF figure of merit", style=general_text_style),
                        html.H5("Filter values from", style=general_text_style_soft),
                        dcc.Input(id='fom_lower_limit_filter_input', type='number', value=0,
                                debounce=True, style=box1_syle),
                        html.H5("to", style=general_text_style_soft),
                        dcc.Input(id='fom_upper_limit_filter_input', type='number', value=1,
                                debounce=True, style=box1_syle),
                ]),

                html.Div([
                        dcc.RangeSlider(id='fom_minmax', min=0, max=1, step=0.05, value=[0,0.5],  marks=None,
                                        tooltip={"placement": "left", "always_visible": True},vertical=True, verticalHeight=100),
                        dcc.Graph(id='fom_graph', figure={}),
                ],style=slider_style_v),                    
            ],style=TwoGraph_style),
            ]),

        ### block-4 (images)
        html.Div([
            html.Br(),
            ### Input Index to select image (or central image?)
            html.Div([
                dcc.Slider(id='image_index', min=1, max=1, step=1, value=1, marks=None,
                           tooltip={"placement": "top", "always_visible": True}),
                dcc.Input(id='image_index_input', type='number', value=1,
                      debounce=True, style=box2_syle) # index box
            ], style=slider_style_m),
            html.Br(),
            html.Div([
                # Keep 1152 x 818 aspect ratio
                dcc.Graph(id='mic_png', figure={}, responsive=True, style={"width" : "43vw", "height" : "30vw"}),
                dcc.Graph(id='ctf_png', figure={}, responsive=True, style={"width" : "30vw", "height" : "30vw"}),
                ], style={"display": "grid", "grid-template-columns": "50% 50%"})
            ],style=OneGraph_style),
])


### Callbacks (take component ID and properties and connect them)

## Callback 1: dropdown job list. Take jobs names from default_pipeline.star
@app.callback(
    [Output(component_id= "import_name", component_property="options"), 
    Output(component_id= "motion_name", component_property="options"),
    Output(component_id= "ctf_name", component_property="options"),
    Output(component_id= "ice_name", component_property="options")],            
    [Input(component_id='interval-component', component_property='n_intervals')])
def load_jobs_name(interval):
    import_jobs= get_jobs_list("default_pipeline.star","import.movies") #temp solution
    motion_jobs= get_jobs_list("default_pipeline.star","motioncorr.own") #temp solution
    ctf_jobs= get_jobs_list("default_pipeline.star","ctffind.ctffind4") #temp solution
    ice_jobs= get_jobs_list("default_pipeline.star","external")
    return (import_jobs, motion_jobs, ctf_jobs, ice_jobs)

## Callback 2: set dimensions of x axis
@app.callback(
    [Output(component_id='xrange', component_property='max')],
    [Input(component_id='import_name', component_property='value'),
     Input(component_id='interval-component', component_property='n_intervals')]
)
def load_import_data(import_name, n_intervals):

    # Get starfile path
    importstar_path = (relion_wd)+'/Import/'+(import_name)+'/movies.star'

    # Check if starfile exists
    if os.path.exists(importstar_path):
        import_df = starfile.read(importstar_path)['movies']
        xrangemax = len(import_df)
    else:
        xrangemax = 0
    return (xrangemax,)

#### The following four functions load graphs data checking from where the callback has been triggered. It is verified by which way the callback 
#### was activated. In case it is by the first run or by a dataset change, the file is read. In any other case (updating of the graph axes), the 
#### data that were saved with (marked as State in the callback annotation) are retaken and the figure is updated without reading the file.

## Callback 3: MotionCorr  
@app.callback(
    [Output(component_id='filter-motion-data-store', component_property='data'),
     Output(component_id='totalmotion_minmax', component_property='max'),
     Output(component_id='totalmotion_graph', component_property='figure'),
     Output(component_id='motion_len', component_property='data')],
    [Input(component_id='filter-motion-data-store', component_property='data'),
     Input(component_id='filter-ctf-data-store', component_property='data'),
     Input(component_id='filter-ice-data-store', component_property='data'),
     Input(component_id='motion_name', component_property='value'),
     Input(component_id='xrange', component_property='value'),
     State(component_id='totalmotion_minmax', component_property='max'),
     Input(component_id='totalmotion_minmax', component_property='value'),
     Input(component_id='motion_upper_limit_filter_input', component_property='value'),
     Input(component_id='motion_lower_limit_filter_input', component_property='value'),
     State(component_id='totalmotion_graph', component_property='figure'),
     State(component_id='motion_len', component_property='data'),
     Input(component_id='interval-component', component_property='n_intervals')]
)
def load_motion_data(motionfilterdata, ctffilterdata, icefilterdata, motion_name, xrange, input_totalmotion_max, input_totalmotion_minmax, motion_upper_limit_filter, motion_lower_limit_filter, input_totalmotion_graph, input_motion_len, n_intervals):

    component_id = ctx.triggered_id

    if component_id in [None, 'motion_name', 'filter-motion-data-store','interval-component']:

        # Get starfile path
        motionstar_path = (relion_wd)+'/MotionCorr/'+(motion_name)+'/corrected_micrographs.star'
        # Check if starfile exists
        if os.path.exists(motionstar_path):
            motion_df = starfile.read(motionstar_path)['micrographs']
            motion_df.index += 1
            motion_df['UserFilterMotion'] = (motion_df['rlnAccumMotionTotal'] >= motion_lower_limit_filter) & (motion_df['rlnAccumMotionTotal'] <= motion_upper_limit_filter) # Adding a column to filter out using user-provided limits
            # clean_motion_df = motion_df[(motion_df['UserFilterMotion'])]
            motiondataout = json.dumps(motion_df.to_dict())

            motion_df = merge_filters(motion_df, ctffilterdata, 'UserFilterMotion', 'UserFilterCtf')
            motion_df = merge_filters(motion_df, icefilterdata, 'UserFilterMotion', 'UserFilterIce')

            motionmax = max(motion_df['rlnAccumMotionTotal'])+10
            motion_len = len(motion_df)
            totalmotion_graph = plot_scatter(motion_df, input_totalmotion_minmax, xrange, 'Total Motion (\u212B)', 'rlnAccumMotionTotal', color_motion1, 'UserFilterMotion',motion_upper_limit_filter, motion_lower_limit_filter)
            totalmotion_graph.update_layout(uirevision=motion_name)
        else:
            motionmax = 0
            motion_len = 0
            totalmotion_graph = empty_graph("No motion data loaded")
            motiondataout = None
    else:
        totalmotion_graph = go.Figure(input_totalmotion_graph)
        if component_id == "totalmotion_minmax":
            totalmotion_graph.update_yaxes(range = input_totalmotion_minmax, autorange=False)
        elif component_id == "xrange":
            totalmotion_graph.update_xaxes(range = xrange, autorange=False)
        motion_len = input_motion_len
        motionmax = input_totalmotion_max
        motiondataout = motionfilterdata

    return (motiondataout, motionmax, totalmotion_graph, motion_len)

## Callback 4: CTF 
@app.callback(
    [Output(component_id='filter-ctf-data-store', component_property='data'),
     Output(component_id='defocus_minmax', component_property='max'),
     Output(component_id='defocus_graph', component_property='figure'),
     Output(component_id='astigmatism_minmax', component_property='max'),
     Output(component_id='astigmatism_graph', component_property='figure'),
     Output(component_id='maxres_minmax', component_property='max'),
     Output(component_id='maxres_graph', component_property='figure'),
     Output(component_id='fom_minmax', component_property='max'),
     Output(component_id='fom_graph', component_property='figure'),
     Output(component_id='image_index', component_property='max')
    ],
    [Input(component_id='filter-motion-data-store', component_property='data'),
     Input(component_id='filter-ctf-data-store', component_property='data'),
     Input(component_id='filter-ice-data-store', component_property='data'),
     Input(component_id='ctf_name', component_property='value'),
     Input(component_id='xrange', component_property='value'),
     Input(component_id='defocus_minmax', component_property='value'),
     Input(component_id='astigmatism_minmax', component_property='value'),
     Input(component_id='maxres_minmax', component_property='value'),
     Input(component_id='fom_minmax', component_property='value'),
     State(component_id='defocus_minmax', component_property='max'),
     State(component_id='astigmatism_minmax', component_property='max'),
     State(component_id='maxres_minmax', component_property='max'),
     State(component_id='fom_minmax', component_property='max'),
     Input(component_id='defocus_upper_limit_filter_input', component_property='value'),
     Input(component_id='defocus_lower_limit_filter_input', component_property='value'),
     Input(component_id='astigmatism_upper_limit_filter_input', component_property='value'),
     Input(component_id='astigmatism_lower_limit_filter_input', component_property='value'),
     Input(component_id='fom_upper_limit_filter_input', component_property='value'),
     Input(component_id='fom_lower_limit_filter_input', component_property='value'),
     Input(component_id='maxres_upper_limit_filter_input', component_property='value'),
     Input(component_id='maxres_lower_limit_filter_input', component_property='value'),
     State(component_id='defocus_graph', component_property='figure'),
     State(component_id='astigmatism_graph', component_property='figure'),
     State(component_id='maxres_graph', component_property='figure'),
     State(component_id='fom_graph', component_property='figure'),
     State(component_id='image_index', component_property='max'),
     Input(component_id='interval-component', component_property='n_intervals')]
)
def load_ctf_data(motionfilterdata, ctffilterdata, icefilterdata, ctf_name, xrange, defocus_minmax_values, astigmatism_minmax_values, maxres_minmax_values, fom_minmax_values,input_defocus_max, input_astigmatism_max, input_maxres_max, input_fom_max,defocus_upper_limit_filter,defocus_lower_limit_filter,astigmatism_upper_limit_filter,astigmatism_lower_limit_filter,fom_upper_limit_filter,fom_lower_limit_filter,maxres_upper_limit_filter,maxres_lower_limit_filter, input_defocus_graph, input_astigmatism_graph, input_maxres_graph, input_fom_graph, 
                imput_image_index_max, n_intervals):

    component_id = ctx.triggered_id

    if component_id in [None, 'ctf_name', 'interval-component', 'filter-ctf-data-store']:

        # Get starfile path
        ctfstar_path = (relion_wd)+'/CtfFind/'+(ctf_name)+'/micrographs_ctf.star'

        # Check if starfile exists
        if os.path.exists(ctfstar_path):

                
            ctf_df = starfile.read(ctfstar_path)['micrographs']
            ctf_df.index += 1

            ctf_df['rlnDefocusU'] = round(ctf_df['rlnDefocusU'] * 0.0001, 3)
            ctf_df['rlnCtfAstigmatism'] = round(ctf_df['rlnCtfAstigmatism'] * 0.0001, 3)
            ctf_df['rlnCtfMaxResolution'] = round(ctf_df['rlnCtfMaxResolution'], 1)
            ctf_df['rlnCtfFigureOfMerit'] = round(ctf_df['rlnCtfFigureOfMerit'], 3)

            defocusmax = max(ctf_df['rlnDefocusU']) + 1
            astigmatismmax = max(ctf_df['rlnCtfAstigmatism']) + 0.1
            maxresmax = max(ctf_df['rlnCtfMaxResolution']) + 1
            fommax = max(ctf_df['rlnCtfFigureOfMerit']) + 0.1

            # Adding columns to filter out using user-provided limits
            ctf_df['UserFilterDefocus'] = (ctf_df['rlnDefocusU'] >= defocus_lower_limit_filter) & (ctf_df['rlnDefocusU'] <= defocus_upper_limit_filter)
            ctf_df['UserFilterAstigmatism'] = (ctf_df['rlnCtfAstigmatism'] >= astigmatism_lower_limit_filter) & (ctf_df['rlnCtfAstigmatism'] <= astigmatism_upper_limit_filter)
            ctf_df['UserFilterFigureOfMerit'] = (ctf_df['rlnCtfFigureOfMerit'] >= fom_lower_limit_filter) & (ctf_df['rlnCtfFigureOfMerit'] <= fom_upper_limit_filter)
            ctf_df['UserFilterMaxResolution'] = (ctf_df['rlnCtfMaxResolution'] >= maxres_lower_limit_filter) & (ctf_df['rlnCtfMaxResolution'] <= maxres_upper_limit_filter)

            ctf_df['UserFilterCtf'] = ctf_df['UserFilterDefocus'] & ctf_df['UserFilterAstigmatism'] & ctf_df['UserFilterFigureOfMerit'] & ctf_df['UserFilterMaxResolution']

            ctfdataout = json.dumps(ctf_df.to_dict())

            ctf_df = merge_filters(ctf_df, motionfilterdata, 'UserFilterCtf', 'UserFilterMotion')
            ctf_df = merge_filters(ctf_df, icefilterdata, 'UserFilterCtf', 'UserFilterIce')

            defocus = plot_scatter(ctf_df, defocus_minmax_values, xrange, 'Defocus (\u03BCm)', 'rlnDefocusU', color_ctf1, 'UserFilterCtf', defocus_upper_limit_filter, defocus_lower_limit_filter)
            astigmatism = plot_scatter(ctf_df, astigmatism_minmax_values, xrange, 'Astigmatism (\u03BCm)', 'rlnCtfAstigmatism', color_ctf1, 'UserFilterCtf', astigmatism_upper_limit_filter, astigmatism_lower_limit_filter)
            maxres = plot_scatter(ctf_df, maxres_minmax_values, xrange, 'Max Resolution (\u212B)', 'rlnCtfMaxResolution', color_res1, 'UserFilterCtf', maxres_upper_limit_filter, maxres_lower_limit_filter)
            fom = plot_scatter(ctf_df, fom_minmax_values, xrange, 'Figure Of Merit', 'rlnCtfFigureOfMerit', color_ctf1, 'UserFilterCtf', fom_upper_limit_filter, fom_lower_limit_filter)

            defocus.update_layout(uirevision=ctf_name)
            astigmatism.update_layout(uirevision=ctf_name)
            maxres.update_layout(uirevision=ctf_name)
            fom.update_layout(uirevision=ctf_name)
 
            max_index = len(ctf_df)
        else:
            defocusmax = 6
            astigmatismmax = 0.1
            maxresmax = 60
            fommax = 0.5

            defocus = empty_graph("No CTF data loaded")
            astigmatism = empty_graph("No CTF data loaded")
            maxres = empty_graph("No CTF data loaded")
            fom = empty_graph("No CTF data loaded")

            ctfdataout = None

            max_index = 0
    else:
        defocus = go.Figure(input_defocus_graph)
        astigmatism = go.Figure(input_astigmatism_graph)
        maxres = go.Figure(input_maxres_graph)
        fom = go.Figure(input_fom_graph)

        if component_id == "xrange":
            defocus.update_xaxes(range = xrange, autorange=False)
            astigmatism.update_xaxes(range = xrange, autorange=False)
            maxres.update_xaxes(range = xrange, autorange=False)
            fom.update_xaxes(range = xrange, autorange=False)
        elif component_id == "defocus_minmax":
            defocus.update_yaxes(range = defocus_minmax_values, autorange=False)
        elif component_id == "astigmatism_minmax":
            astigmatism.update_yaxes(range = astigmatism_minmax_values, autorange=False)
        elif component_id == "maxres_minmax":
            maxres.update_yaxes(range = maxres_minmax_values, autorange=False)
        elif component_id == "fom_minmax":
            fom.update_yaxes(range = fom_minmax_values, autorange=False)
        
        defocusmax = input_defocus_max
        astigmatismmax = input_astigmatism_max
        maxresmax = input_maxres_max
        fommax = input_fom_max
        ctfdataout = ctffilterdata

        max_index = imput_image_index_max

    return (ctfdataout, defocusmax, defocus, astigmatismmax, astigmatism, maxresmax, maxres, fommax, fom, max_index)

## Callback 5: ICE 
@app.callback(
    [Output(component_id='filter-ice-data-store', component_property='data'),
     Output(component_id='ice_minmax', component_property='max'),
     Output(component_id='ice_graph', component_property='figure'),
     Output(component_id='ice_len', component_property='data'),
     Output(component_id='filtered-df-export-store', component_property='data'),
     Output(component_id='filtered-df-export-len', component_property='children')],
    [Input(component_id='filter-motion-data-store', component_property='data'),
     Input(component_id='filter-ctf-data-store', component_property='data'),
     Input(component_id='filter-ice-data-store', component_property='data'),
     Input(component_id='ice_name', component_property='value'),
     Input(component_id='xrange', component_property='value'),
     State(component_id='ice_minmax', component_property='max'),
     Input(component_id='ice_minmax', component_property='value'),
     Input(component_id='ice_upper_limit_filter_input', component_property='value'),
     Input(component_id='ice_lower_limit_filter_input', component_property='value'),
     State(component_id='ice_graph', component_property='figure'),
     State(component_id='ice_len', component_property='data'),
     Input(component_id='filtered-df-export-store', component_property='data'),
     Input(component_id='filtered-df-export-len', component_property='value'),
     Input(component_id='interval-component', component_property='n_intervals')]
)
def load_ice_data(motionfilterdata, ctffilterdata, icefilterdata, ice_name, xrange, input_ice_max, input_ice_minmax, ice_upper_limit_filter, ice_lower_limit_filter, input_ice_graph, input_ice_len, input_filtered_df, input_filtered_micrographs_len, n_intervals):
    component_id = ctx.triggered_id

    filtered_micrographs_len = 0
    if component_id in [None, 'ice_name', 'interval-component', 'filter-ice-data-store','filtered-df-export-store']:

        # Get starfile path
        icestar_path = (relion_wd)+'/External/'+(ice_name)+'/micrographs_ctf_ice.star'

        # Check if starfile exists
        if os.path.exists(icestar_path):
            ice_df = starfile.read(icestar_path)['micrographs']
            ice_df.index += 1
            icemax = int(max(ice_df['rlnMicrographIceThickness'])) + 1
            ice_len = len(ice_df)

            ice_df['UserFilterIce'] = (ice_df['rlnMicrographIceThickness'] >= ice_lower_limit_filter) & (ice_df['rlnMicrographIceThickness'] <= ice_upper_limit_filter) # Adding column for user filter
            
            icedataout = json.dumps(ice_df.to_dict())

            ice_df = merge_filters(ice_df, motionfilterdata, 'UserFilterIce', 'UserFilterMotion')
            ice_df = merge_filters(ice_df, ctffilterdata, 'UserFilterIce', 'UserFilterCtf')
            
            ice_graph = plot_scatter(ice_df, input_ice_minmax, xrange, 'Ice Thickness Score', 'rlnMicrographIceThickness', color_ice1, 'UserFilterIce', ice_upper_limit_filter, ice_lower_limit_filter)
            ice_graph.update_layout(uirevision=ice_name)
            clean_ice_df = ice_df[(ice_df['UserFilterIce'])]
            filtered_micrographs_len = len(clean_ice_df)
            clean_df_export = json.dumps(clean_ice_df.to_dict())
        else:
            icemax = 0
            ice_len = 0
            ice_graph = empty_graph("No ice data loaded")
            icedataout = None
            clean_df_export = None
    else:
        ice_graph = go.Figure(input_ice_graph)
        if component_id == "ice_minmax":
            ice_graph.update_yaxes(range = input_ice_minmax, autorange=False)
        elif component_id == "xrange":
            ice_graph.update_xaxes(range = xrange, autorange=False)
        icemax = input_ice_max
        ice_len = input_ice_len
        icedataout = icefilterdata
        clean_df_export = input_filtered_df
        filtered_micrographs_len = input_filtered_micrographs_len


    return (icedataout, icemax, ice_graph, ice_len, clean_df_export, filtered_micrographs_len)

## Callback 6: progress bars 
@app.callback(
    Output(component_id='progress_graph', component_property='figure'),
    [Input(component_id='motion_len', component_property='data'),
     Input(component_id='xrange', component_property='max'),
     Input(component_id='image_index', component_property='max'),
     Input(component_id='ice_len', component_property='data')]
)
def load_progress(motion_len, xrange, ctf_len, ice_len):

    # Plotting progress data
    progress = go.Figure()

    progress.add_trace(go.Indicator(
        domain = {'x': [0.10, 0.30], 'y': [0, 1]},
        value = motion_len,
        mode = "gauge+number+delta",
        title = {'text': "MotionCorr"},
        delta = {'reference': xrange},
        gauge = {'steps': [{'range': [0,xrange], 'thickness':0.4, 'color':color_motion2}], 'axis': {'range': [None, xrange],'visible': False}, 'borderwidth': 0,  'shape': "bullet", 'bar': {'color': color_motion1,'thickness':0.4}},
        ))
    progress.add_trace(go.Indicator(
        domain = {'x': [0.45, 0.65], 'y': [0, 1]},
        value = ctf_len,
        mode = "gauge+number+delta",
        title = {'text': "CtfFind"},
        delta = {'reference': xrange},
        gauge = {'steps': [{'range': [0,xrange], 'thickness':0.4, 'color':color_ctf2}], 'axis': {'range': [None, xrange],'visible': False}, 'borderwidth': 0,'shape': "bullet",  'bar': {'color': color_ctf1,'thickness':0.4}},
        ))
    
    progress.add_trace(go.Indicator(
        domain = {'x': [0.80, 1.0], 'y': [0, 1]},
        value = ice_len,
        mode = "gauge+number+delta",
        title = {'text': "Ice Thickness"},
        delta = {'reference': xrange},
        gauge = {'steps': [{'range': [0,xrange], 'thickness':0.4, 'color':color_ice2}], 'axis': {'range': [None, xrange],'visible': False}, 'borderwidth': 0,'shape': "bullet",  'bar': {'color': color_ice1,'thickness':0.4}},
        ))
    
    progress.update_layout(height=220, )

    return progress
## Callback 7: Select micrograph index 
# This function identifies where the index of the micrograph comes from. With the index, the corresponding micrographs are selected and displayed.
@app.callback(
    [Output(component_id='image_index', component_property='value'), 
    Output(component_id='image_index_input', component_property='value'), 
    Output(component_id='mic_png', component_property='figure'),
    Output(component_id='ctf_png', component_property='figure')],
    [Input(component_id='ctf_name', component_property='value'),
    Input(component_id='image_index', component_property='value'),
    Input(component_id='image_index_input', component_property='value'),
    Input(component_id='totalmotion_graph', component_property='clickData'),
    Input(component_id='defocus_graph', component_property='clickData'),
    Input(component_id='astigmatism_graph', component_property='clickData'),
    Input(component_id='maxres_graph', component_property='clickData'),
    Input(component_id='fom_graph', component_property='clickData'),
    Input(component_id='ice_graph', component_property='clickData')],
)
def select_image(ctf_name, image_index, image_index_input, totalmotion_clickdata, defocus_clickdata, astigmatism_clickdata, maxres_clickdata, fom_clickdata, ice_clickdata, ):
    component_id = ctx.triggered_id
    if 'totalmotion_graph' == component_id:
        point_number = totalmotion_clickdata['points'][0]['x']
    elif 'defocus_graph' == component_id:
        point_number = defocus_clickdata['points'][0]['x']
    elif 'astigmatism_graph' == component_id:
        point_number = astigmatism_clickdata['points'][0]['x']
    elif 'maxres_graph' == component_id:
        point_number = maxres_clickdata['points'][0]['x']
    elif 'fom_graph' == component_id:
        point_number = fom_clickdata['points'][0]['x']
    elif 'ice_graph' == component_id:
        point_number = ice_clickdata['points'][0]['x']
    elif 'image_index_input' == component_id:
        point_number = image_index_input
    else:
        point_number = image_index

    ctfstar_path = (relion_wd)+'/CtfFind/'+(ctf_name)+'/micrographs_ctf.star'

    #Read ctf dataframe and check if file and index are ok 
    if os.path.exists(ctfstar_path) and point_number > 0:
        ctf_df = starfile.read(ctfstar_path)['micrographs']
        ctf_df.index += 1

        mic_file_df = ctf_df['rlnMicrographName']
        ctf_file_df = ctf_df['rlnCtfImage']
        mic_file = mic_file_df[point_number].replace('mrc','png')
        ctf_file = ctf_file_df[point_number].replace('ctf:mrc','png')

        ## Load Images
        try:
            mic_png_src = app.get_asset_url(mic_file)
            mic_image = Image.open(mic_file)
            mic_png = image_to_figure(mic_png_src, mic_image.size[0], mic_image.size[1], 1) #enables to zoom in the image 
        except:
            mic_png = empty_graph("Error, ensure you have run png_out.py")

        try:
            ctf_png_src = app.get_asset_url(ctf_file)
            ctf_image = Image.open(ctf_file)
            ctf_png = image_to_figure(ctf_png_src, ctf_image.size[0], ctf_image.size[1], 1) 
        except:
            ctf_png = empty_graph("Error, ensure you have run png_out.py")
    else:
        mic_png = empty_graph("No micrograph loaded")
        ctf_png = empty_graph("No CTF data loaded")

    return (point_number, point_number, mic_png, ctf_png)



@app.callback(
    [Output(component_id='placeholder', component_property='data')],
    [Input(component_id='ice_name', component_property='value'),
     Input(component_id='filtered-df-export-store', component_property='data'),
     Input(component_id='export-filtered-data-button', component_property='n_clicks'),
     Input(component_id='interval-component', component_property='n_intervals')])

def export_filtered_starfile(ice_name, filtered_data, button_pressed_clicks, n_intervals):
    component_id = ctx.triggered_id
    if (component_id in ['export-filtered-data-button']) | (n_intervals % 2 == 0):
        icestar_path = (relion_wd)+'/External/'+(ice_name)+'/micrographs_ctf_ice.star'
        ReadInMicrographs = pd.DataFrame.from_dict(json.loads(filtered_data)).reset_index(drop=True)
        ReadInOptics = starfile.read(icestar_path)['optics']
        ReadInMerge = {'optics' : ReadInOptics, 'micrographs' : ReadInMicrographs}
        starfile.write(ReadInMerge, 'exported_filtered_micrographs.star', overwrite=True)
    return(['bla'])

#APP Start
if __name__ == '__main__':
    app.run_server(debug=debug_mode, dev_tools_hot_reload = False, use_reloader=True, host=hostname, port=port_number)