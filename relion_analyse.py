#!/usr/bin/env python3

"""
Relion analyse dashboard
Nayim Gonzalez-Rodriguez, Emma Arean-Ulloa & Rafael Fernandez-Leiro 2022
"""

"""
Activate conda environment before running
Usage: run relion_analyse.py in your relion project directory
"""

### Libraries setup
import os
import pandas as pd
import starfile
import dash
from dash import html
from dash import dcc
from dash import ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import socket
import argparse
import glob
import dash_cytoscape as cyto
import regex as re
from PIL import Image

# Load extra layouts for pipeline display
cyto.load_extra_layouts()

# Parsing port number and host
parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", help = "choose port to run the webapp")
parser.add_argument("--host", "-host", help = "choose host to run the webapp")
parser.add_argument("--debug", "-d", help = "launch app in debug mode")
args, unknown = parser.parse_known_args()

# Set localhost and 8051 as host and port by default
if not args.port: port_number = 8051
else: port_number = args.port
if not args.host: hostname = socket.gethostname()
else: hostname = args.host 
if not args.debug: debug_mode = False
else: debug_mode = args.host

### FUNCTION DEFINITIONS ###

# Reading pipeline data
def pipeline_reader1(pipeline_starfile, nodetype):
    pipeline_df = starfile.read(pipeline_starfile)['pipeline_nodes']
    nodes = list(pipeline_df[pipeline_df['rlnPipeLineNodeTypeLabel'].str.contains(str(nodetype), na=False)]['rlnPipeLineNodeName'])
    return nodes

def pipeline_reader2(pipeline_starfile, nodetype):
    pipeline_df = starfile.read(pipeline_starfile)['pipeline_processes']
    nodes = list(pipeline_df[pipeline_df['rlnPipeLineProcessName'].str.contains(str(nodetype), na=False)]['rlnPipeLineProcessName'])
    return nodes

# Plot scatterplot with side violin plots 
def plot_scatter(dataframe, x_data, y_data, coloring):
    plot = px.scatter(
        data_frame = dataframe,
        x = x_data,
        y = y_data,
        marginal_x = "violin",
        marginal_y = "violin",
        color = coloring,
        render_mode = 'webgl',
        template = 'plotly_white',
        opacity = 0.5
        )
    return plot

# Plot scatterplot with side violin plots using WebGL for large datasets
def plot_scattergl(data_frame, x_data, y_data, coloring, colorscale, hist_color):
    plot = go.Figure()
    main_scatter = plot.add_scattergl(
        x = data_frame[x_data],
        y = data_frame[y_data],
        mode = 'markers',
        marker_colorscale=colorscale,
        marker = dict(color=coloring, opacity=0.5, showscale=True),
        fillcolor = 'white'
    )
    side_histogram_x = plot.add_trace(go.Violin(
        x = data_frame[x_data],
        name = x_data,
        yaxis = 'y2',
        marker = dict(opacity=0.5, color=hist_color),
    ))
    side_histogram_y = plot.add_trace(go.Violin(
        y = data_frame[y_data],
        name = y_data,
        xaxis = 'x2',
        marker = dict(opacity=0.5, color=hist_color),
    ))
    plot.layout = dict(xaxis=dict(domain=[0, 0.85], zeroline=True, title=x_data,gridcolor='#CBCBCB'),
                yaxis=dict(domain=[0, 0.85], zeroline=True, title=y_data, gridcolor='#CBCBCB'),
                showlegend=False,
                margin=dict(t=50),
                hovermode='closest',
                bargap=0,
                xaxis2=dict(domain=[0.85, 1], showgrid=True, zeroline=False),
                yaxis2=dict(domain=[0.85, 1], showgrid=True, zeroline=False),
                plot_bgcolor = 'rgba(0,0,0,0)',
    )
    
    def do_zoom(layout, xaxis_range, yaxis_range):
        inds = ((xaxis_range[0] <= data_frame[x_data]) & (data_frame[x_data] <= xaxis_range[1]) &
                (yaxis_range[0] <= data_frame[y_data]) & (data_frame[y_data] <= yaxis_range[1]))
        with plot.batch_update():
            side_histogram_x.x = data_frame[x_data][inds]
            side_histogram_y.y = data_frame[y_data][inds]

    plot.layout.on_change(do_zoom, 'xaxis.range', 'yaxis.range')

    return plot

def empty_graph(text=None, fontsize=20):
    fig = go.Figure()
    if text:
        fig.add_annotation(text=text, showarrow=False, font = dict(size = fontsize))
    fig.update_layout(template="simple_white", showlegend=False, height=220)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def image_to_figure(src, width, height, scale_factor):
    # Create figure
    fig = go.Figure()

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

# Plotting line plots
def plot_line(data_frame, x_data, y_data):
    plot = px.line(
        x = x_data,
        y = data_frame[y_data],
        render_mode = 'webgl',
        template = 'plotly_white',
    )
    return plot

def plot_line_multi(data_frame, x_data, y_data, coloring):
    plot = px.line(
        x = x_data,
        y = data_frame[y_data],
        color = data_frame[coloring],
        render_mode = 'webgl',
        template = 'plotly_white',
    )
    return plot

# Area plots for class distribution over iterations
def plot_area(data_frame):
    plot = px.area(data_frame, template = 'plotly_white')
    return plot

# Heatmap plots for angular distribution
def plot_angdist(data_frame, x_data, y_data, bins, coloring):
    plot = px.density_heatmap(
        data_frame = data_frame,
        x = x_data,
        y = y_data,
        facet_col = coloring,
        facet_col_wrap = 4,
        color_continuous_scale='deep', #colorscales giving decent contrast: RdBu, deep, rainbow 
        nbinsx = bins,
        nbinsy = bins
    )
    return plot

# Empty graph to show when no data is loaded
def empty_graph(text=None, fontsize=20):
    fig = go.Figure()
    if text:
        fig.add_annotation(text=text, showarrow=False, font = dict(size = fontsize))
    fig.update_layout(template="simple_white", showlegend=False, height=220)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

### STYLE ###

# Header
header_style = {'width':'100%', 'vertical-align':'center' , 'display':'inline-flex', 'justify-content': 'space-between'}
title1_style = {"margin-left": "15px", "margin-top": "15px", "margin-bottom": "0em", "color": "Black",
                "font-family" : "Helvetica", "font-size":"2.5em"}
title2_style = {"margin-left": "15px", "margin-top": "0em", "color": "Black", "font-family" : "Helvetica"}
header_button_style = {'margin-top':'40px','margin-right':'40px'}

# Tabs
pre_style = {'overflowY': 'scroll', 'height': '500px'}
tabs_style = {'height': '3em', 'width': '100%', 'display': 'inline', 'vertical-align': 'bottom', 'borderBottom':'3px #000000'}
tab_style = {'padding':'0.5em', "font-family" : "Helvetica", 'background-color':'white'}
tab_selected_style = {'padding':'0.5em', 'borderTop': '3px solid #000000', "font-family" : "Helvetica", 'font-weight':'bold'}
tab_left_div_style = {'width':'20%', 'vertical-align':'top' , 'display':'inline-block'}
tab_right_div_style = {'width':'80%', 'vertical-align':'top' , 'display':'inline-block'}
tab_bottom_right_style = {'width':'100%', 'vertical-align':'top' , 'display':'inline-block'}
H5_title_style = {'font-family':'Helvetica', 'font-weight':'regular'}

# Dropdowns
dd_style = {"font-size":"0.9em",'width':"100%", "margin-left": "0%", "color": "black",
            "font-family" : "Helvetica", 'vertical-align': 'top', "margin-bottom":"2px"}
box_style = {"font-size":"0.9em",'padding':'0.3em 1.2em','width':"87%", "margin-left": "0%", "color": "black",
            "font-family" : "Helvetica", 'vertical-align': 'center', "margin-bottom":"2px"}

# Buttons
bt_style = {"align-items": "center", "background-color": "#F2F3F4", "border": "2px solid #000",
            "box-sizing": "border-box", "color": "#000", "cursor": "pointer", "display": "inline-flex",
            "font-family": "Helvetica", "margin-bottom":"3px", 'padding':'0.3em 1.2em', 'margin':'0 0.3em 0.3em 0',
            "font-size": "0.9em", 'font-weight':'500', 'border-radius':'2em', 'text-align':'center',
            'transition':'all 0.2s'}

# Pipeline nodes
nodes_style = [
    {'selector': 'node', 'style': {'content': 'data(label)', 'font-family': 'monospace', 'text-wrap': 'wrap',
                                    'text-max-width': '80px', 'text-overflow-wrap': 'anywhere',
                                    'text-halign': 'center', 'text-valign': 'center', 'color': 'white',
                                    'text-outline-width': 2, 'width': '90px', 'height': '90px'}},
    {'selector': '.Import', 'style': {'background-color': '#FFB3D9', 'shape': 'rectangle'}},  # Pastel Pink
    {'selector': '.MotionCorr', 'style': {'background-color': '#FFD9B3'}},  # Pastel Orange
    {'selector': '.CtfFind', 'style': {'background-color': '#C9C9C9'}},  # Silver
    {'selector': '.AutoPick', 'style': {'background-color': '#A3D8E9'}},  # Pastel Blue
    {'selector': '.Extract', 'style': {'background-color': '#FFB3D9', 'shape': 'square'}},  # Pastel Pink
    {'selector': '.Select', 'style': {'background-color': '#B9E5C0', 'width': '120px', 'height': '40px'}},  # Pastel Green
    {'selector': '.Class2D', 'style': {'background-color': '#FFEAB3'}},  # Pastel Yellow
    {'selector': '.Class3D', 'style': {'background-color': '#FFB3B0'}},  # Pastel Salmon
    {'selector': '.Refine3D', 'style': {'background-color': '#D9C3D9'}},  # Thistle
    {'selector': '.MaskCreate', 'style': {'background-color': '#FFB3B0'}},  # Pastel Salmon
    {'selector': '.Polish', 'style': {'background-color': '#C2E0E2'}},  # Pastel Cyan
    {'selector': '.LocalResolution', 'style': {'background-color': '#F0CB95'}},  # Pastel Apricot ## Double check this
    {'selector': '.CtfRefinement', 'style': {'background-color': '#B1E4B4'}},  # Pastel Green ## Double check this
    {'selector': '.Subtract', 'style': {'background-color': '#A8C3D9'}},  # Pastel Blue  # Double check this
    {'selector': '.PostProcess', 'style': {'background-color': '#E9E9E9', 'line-color': '#E9E9E9', 'shape': 'star'}},  # Pastel Gray
    {'selector': '.External', 'style': {'background-color': '#C5E5F0'}},  # Pale Cyan
]


hover_style = {
    'position': 'fixed',
    'z-index': '1',
    'bottom': '10px',
    'right': '10px',
    'padding': '20px',
    'overflow-y': 'scroll',
    'background-color': '#f7f7f7',
    'box-shadow': '0px 2px 5px rgba(0, 0, 0, 0.2)',
    'border-radius': '12px', 
    'height': '40%',
    'width': '20%',
    'font-family': 'Arial, sans-serif',
    'font-size': '1em',
}


job_specific_data = {
                    "relion.import.movies":['Cs', 'Q0', 'angpix', 'beamtilt_x', 'beamtilt_y', 'kV'], 

                    "relion.motioncorr.own":['bfactor', 'bin_factor', 'do_float16', 'dose_per_frame', 'fn_gain_ref', 'gain_flip', 'gain_rot', 'group_frames', 'patch_x', 'patch_y'],

                    "relion.motioncorr.motioncor2":['bfactor', 'bin_factor', 'do_float16', 'dose_per_frame', 'fn_gain_ref', 'gain_flip', 'gain_rot', 'group_frames', 'patch_x', 'patch_y'],

                    "relion.ctffind.ctffind4":['box', 'ctf_win', 'dast', 'dfmax', 'dfmin', 'dfstep', 'use_noDW'],

                    "relion.ctffind.gctf":['box', 'ctf_win', 'dast', 'dfmax', 'dfmin', 'dfstep', 'use_noDW'],

                    "relion.manualpick":['color_label', 'diameter'],

                    "relion.autopick.log":['log_adjust_thr', 'log_diam_max', 'log_diam_min', 'log_invert', 'log_maxres', 'log_upper_thr'],
                    
                    "relion.autopick.topaz.train":['topaz_train_parts','topaz_train_picks'],
                    
                    "relion.autopick.topaz.pick":['topaz_model', 'topaz_nr_particles', 'topaz_other_args', 'topaz_particle_diameter'],
                    
                    "relion.autopick.ref2d":['ref3d_sampling', 'ref3d_symmetry', 'shrink', 'threshold_autopick', ],

                    "relion.extract":['do_fom_threshold', 'do_invert', 'do_recenter', 'do_reextract', 'do_rescale','extract_size', 'minimum_pick_fom', 'rescale'],

                    "relion.extract.reextract":['do_fom_threshold', 'do_invert', 'do_recenter', 'do_reextract', 'do_rescale','extract_size', 'minimum_pick_fom', 'rescale'],

                    "relion.extract":['do_float16', 'do_recenter', 'do_reextract', 'extract_size', 'minimum_pick_fom', 'rescale'],

                    "relion.class2d":[ 'do_center', 'do_em', 'do_grad', 'do_helix', 'highres_limit', 'min_dedicated', 'nr_classes', 'nr_iter_em', 'nr_iter_grad', 'particle_diameter', 'tau_fudge'],

                    "relion.initialmodel":['nr_classes', 'nr_iter', 'particle_diameter','sym_name'],

                    "relion.class3d":['do_ctf_correction', 'do_zero_mask', 'do_fast_subsets', 'do_helix', 'do_local_ang_searches', 'dont_skip_align', 'highres_limit', 'ini_high', 'nr_classes', 'nr_iter', 'particle_diameter', 'range_psi', 'range_rot', 'range_tilt', 'sigma_angles', 'sym_name', 'tau_fudge'],

                    "relion.refine3d":['auto_faster', 'auto_local_sampling', 'do_ctf_correction', 'do_zero_mask', 'ini_high', 'other_args', 'particle_diameter', 'range_psi', 'range_rot', 'range_tilt', 'sampling', 'sym_name'],

                    "relion.external":['fn_exe', 'in_3dref', 'in_coords', 'in_mask', 'in_mic', 'in_mov', 'in_part','param1_label', 'param1_value', 'param2_label', 'param2_value', 'param3_label', 'param3_value', 'param4_label', 'param4_value', 'param5_label', 'param5_value', 'param6_label', 'param6_value', 'param7_label', 'param7_value', 'param8_label', 'param8_value', 'param9_label', 'param9_value'],

                    "relion.localres.own":['do_queue',  'fn_mask', 'fn_mtf', 'maxres', 'minres',  'stepres'],

                    "relion.ctfrefine":['do_4thorder', 'do_aniso_mag', 'do_astig', 'do_bfactor', 'do_ctf', 'do_defocus', 'do_phase', 'do_tilt', 'do_trefoil', 'minres'],

                    "relion.ctfrefine.anisomag":['do_4thorder', 'do_aniso_mag', 'do_astig', 'do_bfactor', 'do_ctf', 'do_defocus', 'do_phase', 'do_tilt', 'do_trefoil', 'minres'],
                     
                    "relion.polish":['do_polish', 'do_float16', 'do_own_params', 'do_param_optim', 'eval_frac', 'extract_size', 'first_frame', 'last_frame', 'maxres', 'min_dedicated', 'minres', 'opt_params', 'optim_min_part', 'rescale', 'sigma_acc', 'sigma_div', 'sigma_vel'],

                    "relion.polish.train":['do_polish', 'do_float16', 'do_own_params', 'do_param_optim', 'eval_frac', 'extract_size', 'first_frame', 'last_frame', 'maxres', 'min_dedicated', 'minres', 'opt_params', 'optim_min_part', 'rescale', 'sigma_acc', 'sigma_div', 'sigma_vel'],

                    "relion.postprocess":['adhoc_bfac', 'angpix', 'autob_lowres', 'do_adhoc_bfac', 'do_auto_bfac','do_skip_fsc_weighting', 'low_pass', 'min_dedicated'],

                    "relion.maskcreate":['angpix', 'extend_inimask', 'inimask_threshold', 'lowpass_filter', 'width_mask_edge']


                     }


### Project directory

relion_wd = os.getcwd()

print('starting up relion_live dashboard in ' + str(relion_wd) + ' ...')

### Initialising dash APP ###

assets_path = relion_wd # this reads the whole folder (!!) so takes long if it is a big project 

app = dash.Dash(
    __name__,
    assets_folder=assets_path,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "RELION Analyse Dashboard"
server = app.server

### APP Layout ###

def serve_layout():
    return html.Div([
        # Title
        
        html.Div(style=header_style,children=[
            html.Div(children=[
                html.H1("RELION", style=title1_style),
                html.H2("Analyse Dashboard", style=title2_style),
                ]),
            html.Div(style=header_button_style, children=[
                html.A(html.Button('Reload pipeline', style=bt_style), id='RefreshPage', href='/'),
                ]),
            ]),

        # Tabs

        html.Div([
            dcc.Tabs([
                # Tab Pipeline

                dcc.Tab(label=' Relion Pipeline', children=[
                    html.Div(style={'width':'10%'}, children=[
                        # Reload Graph button
                    html.A(html.Button('Reload graph', style=bt_style), id='reloadgraphbutton', href='/'),
                        ]),

                    html.Div(children=[   
                        html.Div(style={'width':'100%'},children=[                    
                            # Graphs
                            cyto.Cytoscape(id='pipeline_graph', layout={'name': 'dagre'},
                                           style={'width': '100%', 'height': '800px'},
                                           boxSelectionEnabled=True,
                                           elements=[],
                                           stylesheet=nodes_style),
                            html.Div(id='hovered_node_info', style=hover_style),
                        ]),
                    ]),
                    ], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse Micrographs

                dcc.Tab(label=' Analyse Micrographs', children=[
                    html.Div(style=tab_left_div_style, children=[
                        # Dropdowns for starfile selection
                        html.H5('Micrograph starfile to analyse', style=H5_title_style),
                        dcc.Dropdown(id='mic_star', placeholder='choose starfile...', options=pipeline_reader1('default_pipeline.star', 'MicrographsData'), style=dd_style),
                        # Dropdowns for variable selection
                        html.H5('Choose axes (x, y and colouring)', style=H5_title_style),
                        html.Div(children=[
                            dcc.Dropdown(id='mic_dropdown_x', value='Index', options=[], style=dd_style),
                            dcc.Dropdown(id='mic_dropdown_y', value='rlnCtfMaxResolution', options=[], style=dd_style),
                            dcc.Dropdown(id='mic_dropdown_class', value='rlnCtfFigureOfMerit', options=[], style=dd_style),
                        ]),

                        # Scatter plot actions
                        html.H5("SCATTER PLOT", style=H5_title_style),    

                        # Selected micrographs print and export button
                        html.H5("Export selection based on SCATTER plot (basename):", style={'font-family':'Helvetica', 'font-weight':'regular'}),
                        html.Div(style={'width':'90%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'40%'}, children=[
                                dcc.Input(id='basename_export_mic', value='export', style=box_style),
                            ]),
                            html.Div(style={'width':'20%'}, children=[
                                html.Button('Export', id='export_micrographs_button', style=bt_style),
                            ]),
                            html.Div(style={'width':'20%'}, children=[
                                html.Button('Display', id='display_sel_mic', style=bt_style),
                            ]),
                        ]),
                        # Selected micrographs list                    
                        html.Div(style={'width':'95%','margin-right':'1%','margin-left':'1%'},children=[                    
                            html.Pre(id='selected_micrographs' , style=pre_style),
                        ]),
                    ]),
        
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'100%'},children=[                    
                            # Graphs
                            dcc.Graph(id='mic_scatter2D_graph', figure={}, style={'display': 'inline-block', 'width': '80%', 'height': '50vh'}),
                            # Mic-CTF pngs
                            html.Div([
                                dcc.Graph(id='mic_png', figure={}, responsive=True, style={"width" : "28vw", "height" : "20vw"}),
                                dcc.Graph(id='ctf_png', figure={}, responsive=True, style={"width" : "20vw", "height" : "20vw"}),
                                ], style={"display": "grid", "grid-template-columns": "50% 50%", "justify-items": "start"})
                            ]),
                        ]),
                    ], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse Particles

                dcc.Tab(label=' Analyse Particles', children=[
                    html.Div(style=tab_left_div_style,children=[
                        # Dropdowns for starfile selection
                        html.H5('Particle starfile to analyse', style=H5_title_style),
                        dcc.Dropdown(id='ptcl_star', placeholder='choose starfile...', options=pipeline_reader1('default_pipeline.star', 'ParticlesData'), style=dd_style),
            
                        # Dropdowns for variable selection
                        html.H5('Choose axes (x, y and colouring)', style={'font-family':'Helvetica', 'font-weight':'regular'}),
                        html.Div(children=[
                            dcc.Dropdown(id='ptcl_dropdown_x', value='rlnDefocusU', options=[], style=dd_style),
                            dcc.Dropdown(id='ptcl_dropdown_y', value='rlnDefocusV', options=[], style=dd_style),
                            dcc.Dropdown(id='ptcl_dropdown_class', value='rlnOpticsGroup', options=[], style=dd_style),
                        ]),

                        # Selected particles print and export button
                        html.H5("Export selection (basename):", style=H5_title_style),
                        html.Div(style={'width':'90%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'40%'}, children=[
                                dcc.Input(id='basename_export_ptcl', value='exported', style=box_style),
                            ]),
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Export', id='export_particles_button', style=bt_style),
                            ]),
                        ]),
                        # Selected particles list                    
                        html.Div(style={'width':'95%','margin-right':'1%','margin-left':'1%'},children=[                    
                            html.Pre(id='selected_particles' , style=pre_style),
                        ]),    
                    ]),
        
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'100%'},children=[                    
                            # Graphs
                            dcc.Graph(id='ptcl_scatter2D_graph', figure={}, style={'display': 'inline-block', 'width': '80%', 'margin-lef':'15px', 'height': '70vh'}),
                        ]),
                    ]),

                    ], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse 2D Classification

                dcc.Tab(label=' Analyse 2D Classification', children=[
                    html.Div(style=tab_left_div_style,children=[
                        # Dropdowns
                        html.H5('Classification job to analyse', style=H5_title_style),
                        dcc.Dropdown(id='C2Djob2follow', placeholder='choose job to follow...', options=pipeline_reader2('default_pipeline.star', 'Class2D'), style=dd_style),
                        dcc.Input(id='C2Dfollow_msg', type='text', debounce=True, style=box_style),
                        html.H5('Select variable to plot', style=H5_title_style),
                        dcc.Dropdown(id='C2Dfollow_dropdown_y', value='rlnChangesOptimalClasses', options=[], style=dd_style),
                        # Buttons
                        html.H5("Display last iteration (external):", style=H5_title_style),
                        html.Div(style={'width':'100%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display classes (RELION)', id='C2Ddisplay_last_ite', style=bt_style),
                            ]),
                        ]),
                    ]),
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'50%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                            dcc.Graph(id='C2Dfollow_graph', figure={}),
                        ]),
                        html.Div(style={'width':'50%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow classes
                            dcc.Graph(id='C2Dclassnumber_graph', figure={}),
                        ]),
                    ])], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse 3D Classification

                dcc.Tab(label=' Analyse 3D Classification', children=[
                    html.Div(style=tab_left_div_style,children=[
                        # Dropdowns
                        html.H5('Classification job to analyse', style=H5_title_style),
                        dcc.Dropdown(id='job2follow', placeholder='choose job to follow...', options=pipeline_reader2('default_pipeline.star', 'Class3D'), style=dd_style),
                        html.H5('Select variable to plot', style=H5_title_style),
                        dcc.Input(id='follow_msg', type='text', debounce=True, style=box_style),
                        dcc.Dropdown(id='follow_dropdown_y', value='rlnChangesOptimalClasses', options=[], style=dd_style),
                        dcc.Dropdown(id='follow_model_dropdown_y', value='rlnSpectralOrientabilityContribution', options=[], style=dd_style),

                        # Buttons
                        html.H5("Display last iteration (external):", style=H5_title_style),
                        html.Div(style={'width':'100%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display classes (RELION)', id='display_last_ite', style=bt_style),
                            ]),
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display maps (Chimera)', id='display_chimera_last_ite', style=bt_style),
                            ]),
                        ]),
                    ]),
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                            dcc.Graph(id='follow_graph', figure={}),
                        ]),
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow classes
                            dcc.Graph(id='classnumber_graph', figure={}),
                        ]),
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                        # Plot FSC
                        dcc.Graph(id='follow_model', figure={})
                        ]),

                    ]),
                    html.Div(style=tab_bottom_right_style,children=[
                        html.Div(style={'width':'100%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                            dcc.Graph(id='angdist_per_class', figure={}),
                        ]),
                    ])], style=tab_style, selected_style=tab_selected_style),

                # Tab Analyse 3D Refinement

                dcc.Tab(label=' Analyse 3D Refinement', children=[
                    html.Div(style=tab_left_div_style,children=[
                        # Dropdowns
                        html.H5('Refine3D job to analyse', style=H5_title_style),
                        dcc.Dropdown(id='ref_job2follow', placeholder='choose job to follow...', options=pipeline_reader2('default_pipeline.star', 'Refine3D'), style=dd_style),
                        html.H5('Select variable to plot', style=H5_title_style),
                        dcc.Input(id='ref_follow_msg', type='text', debounce=True, style=box_style),
                        dcc.Dropdown(id='ref_follow_dropdown_y', value='rlnCurrentResolution', options=[], style=dd_style),
                        dcc.Dropdown(id='ref_follow_model_dropdown_y', value='rlnGoldStandardFsc', options=[], style=dd_style),
                        # Buttons
                        html.H5("Display last iteration (external):", style=H5_title_style),
                        html.Div(style={'width':'100%', 'vertical-align':'center', 'display':'flex', 'justify-content': 'space-between'}, children=[
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display slices (RELION)', id='ref_display_last_ite', style=bt_style),
                            ]),
                            html.Div(style={'width':'50%'}, children=[
                                html.Button('Display maps (Chimera)', id='ref_display_chimera_last_ite', style=bt_style),
                            ]),
                        ]),
                    ]),
                    html.Div(style=tab_right_div_style,children=[
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                        dcc.Graph(id='ref_follow_graph', figure={}),
                        ]),
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot FSC
                        dcc.Graph(id='ref_follow_model', figure={})
                        ]),
                        html.Div(style={'width':'33%', 'vertical-align':'top' , 'display':'inline-block'},children=[
                            # Plot follow progression
                        dcc.Graph(id='ref_follow_angdist', figure={}),
                        ]),

                    ])], style=tab_style, selected_style=tab_selected_style),

                ], style=tabs_style),
        ]),
    ])

app.layout = serve_layout

### Callbacks

## Callback reload pipeline
@app.callback(
    [Output(component_id='RefreshPage', component_property='title')],
    [Input(component_id='RefreshPage', component_property='n_clicks')],
    prevent_initial_call=True)

def reload_pipeline(pipeline_reload_button_press):

    pipeline_reload_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'pipeline_reload_button' in pipeline_reload_button_press_changed:
        print('reloading pipeline')
        bttitle = ''
        
    return ([bttitle])

## Callback Pipeline Graph
@app.callback(
    [Output(component_id='pipeline_graph', component_property='elements'),
     Output(component_id='hovered_node_info', component_property='children'),
     Output(component_id='hovered_node_info', component_property='style')],
    [Input(component_id='reloadgraphbutton', component_property='n_clicks'),
     Input(component_id='pipeline_graph', component_property='tapNodeData'),
     State(component_id='pipeline_graph', component_property='elements')],

    prevent_initial_call=False)


def plot_pipeline(reloadgraphbutton, tapdata, current_cytoelements):

    # If callback has been triggered by reaload button or is the first launch,
    # then read pipeline data and create Cytoscape graph
    if 'pipeline_graph' != ctx.triggered_id:

        print('reloading graph')

        # Read default_pipeline starfile
        pipe_nodes = starfile.read('default_pipeline.star')['pipeline_processes']['rlnPipeLineProcessName']
        pipe_input_edges = starfile.read('default_pipeline.star')['pipeline_input_edges']

        cytoelements = []

        for i in pipe_nodes:
            cytoelements.append({'data': {'id':i[-7:-1], 'label':i}, 'classes':i[:-8]})

        for i, row in pipe_input_edges.iterrows():

            try: cytoelements.append({'data': {'source':re.search(r'job\d\d\d', row['rlnPipeLineEdgeFromNode'])[0], 'target':row['rlnPipeLineEdgeProcess'][-7:-1]}})

            except:

                print('there\'s a problem with node: '+row['rlnPipeLineEdgeFromNode'])

                continue

        job_data_output_ = "No node selected, nothing to display"
        hoverstyle = None  


    # Otherwise, show clicked job's information
    else:

        cytoelements = current_cytoelements

        try:

            job_file = str(tapdata['label'])+str('job.star')

            print(job_file)
            job_data = starfile.read(job_file)
            job_type = job_data['job']['rlnJobTypeLabel'][0]


            with open(job_file, 'r') as file: job_data_output = file.read()

            #with starfile.read(job_file) as ffile: print(ffile)

            job_data_output_ = []

            job_data_output_.append(job_file)
            job_data_output_.append(html.Br())
            job_data_output_.append(html.Br())

            for i in job_specific_data[job_type]:
                val = job_data['joboptions_values']['rlnJobOptionValue'][job_data['joboptions_values']['rlnJobOptionVariable'] == i].to_string(index=False)
                line = str(i) + ':\t' + val
                job_data_output_.append(line)
                job_data_output_.append(html.Br())

            hoverstyle = hover_style

        except:

            job_data_output_ = "No node selected, nothing to display"

            hoverstyle = None   

    return ([cytoelements, job_data_output_, hoverstyle])


## Callback Micrographs
@app.callback(
    [Output(component_id='mic_dropdown_x', component_property='options'),
     Output(component_id='mic_dropdown_y', component_property='options'),
     Output(component_id='mic_dropdown_class', component_property='options'),
     Output(component_id='mic_scatter2D_graph', component_property='figure'),
     Output(component_id='selected_micrographs', component_property='children')
    ],
    [Input(component_id='mic_star', component_property='value'),
     Input(component_id='mic_dropdown_x', component_property='value'),
     Input(component_id='mic_dropdown_y', component_property='value'),
     Input(component_id='mic_dropdown_class', component_property='value'),
     State(component_id='mic_scatter2D_graph', component_property='selectedData'),
     State(component_id='basename_export_mic', component_property='value'),
     Input(component_id='export_micrographs_button', component_property='n_clicks'),
     Input(component_id='display_sel_mic', component_property='n_clicks'),
     ]
)

def load_df_and_graphs(mic_star, mic_dd_x, mic_dd_y, mic_dd_class,
                       selectedMicData, basename_export_mic,
                       exportMic_button_press, display_sel_mic_button_press):

### Micrographs

    # Importing CTF starfile data
    try:
        ctf_df_optics = starfile.read(mic_star)['optics']
        ctf_df = starfile.read(mic_star)['micrographs']
    except:
        print('No valid starfile selected')
        raise PreventUpdate

    # Importing MotionCorr data from all CTF-corrected micrographs, even if they
    # come from different motioncor jobs.
    motion_df = ''
    for i in ctf_df['rlnMicrographName'].str[:18].unique():
        motion_star_path = str(i)+'/corrected_micrographs.star'
        try:
            if type(motion_df) == type('') :
                motion_df = starfile.read(motion_star_path)['micrographs']
            else:
                motion_df = pd.concat([motion_df , starfile.read(motion_star_path)['micrographs']], ignore_index = True)
        except:
            print("motion_df empty, can't find the files")

    # Merging CTF and MotionCorr data in a single dataframe for easy plotting
    if type(motion_df) == pd.DataFrame:
        motion_df = motion_df.drop('rlnOpticsGroup', axis=1)
        job_df = pd.merge(ctf_df, motion_df).drop(columns=['rlnMicrographName', 'rlnCtfPowerSpectrum', 'rlnMicrographMetadata']) # what if they don't match?
        job_df.insert(0, 'Index', job_df.index)
    else:
        job_df = ctf_df
        print("motion_df empty, can't find the files")

    mic_dropdown_x = list(job_df)
    mic_dropdown_y = list(job_df)
    mic_dropdown_class = list(job_df)
   
    # Duplicating df info (really needed?)
    mic_dff = job_df.copy()

    # Color definitions
    color1 = 'cornflowerblue'

    if mic_dd_class == None: mic_dd_class = 'cornflowerblue'
    else: mic_dd_class = mic_dff[mic_dd_class].astype(float)

    # Plot using WebGL given probably large dataframes
    mic_scatter2D = plot_scattergl(mic_dff, mic_dd_x, mic_dd_y, mic_dd_class, 'Viridis', 'cornflowerblue')
    mic_scatter2D.update_layout()

    # Parsing info from manual on-plot selection
    selected_micrographs_indices = []
    NOTselected_micrographs_indices = []
    if isinstance(selectedMicData, dict):
        print("selecting based on scatter plot selection")
        for i in selectedMicData['points']:
            selected_micrographs_indices.append(int(i['pointIndex']))
        NOTselected_micrographs = mic_dff.loc[mic_dff.index.difference(selected_micrographs_indices)]
        NOTselected_micrographs_indices = list(NOTselected_micrographs.index.values)

    # Output definitions
    selectionMic_output = 'You\'ve selected '+str(len(selected_micrographs_indices))+' micrographs on SCATTER plot with indices: '+str(selected_micrographs_indices)
    outfile_mic_YES = str(basename_export_mic + '_selected_micrographs.star')
    outfile_mic_NO = str(basename_export_mic + '_not_selected_micrographs.star')
    exportMic_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'export_micrographs_button' in exportMic_button_press_changed:
        dict_mic_output_YES = {'optics' : ctf_df_optics , 'micrographs' : ctf_df.iloc[selected_micrographs_indices]}
        dict_mic_output_NO = {'optics' : ctf_df_optics , 'micrographs' : ctf_df.iloc[NOTselected_micrographs_indices]}
        starfile.write(dict_mic_output_YES, outfile_mic_YES, overwrite=True)
        starfile.write(dict_mic_output_NO, outfile_mic_NO, overwrite=True)
        print('Exported selected micrographs as '+ outfile_mic_YES + ' and not selected micrographs as ' + outfile_mic_NO)

    # Display mics
    display_sel_mic_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_sel_mic' in display_sel_mic_button_press_changed:
        os.system(str('`which relion_display` --i '+outfile_mic_YES+' --gui '))

    ### RETURN

    return (mic_dropdown_x, mic_dropdown_y, mic_dropdown_class, mic_scatter2D, selectionMic_output)

@app.callback(
    [Output(component_id='mic_png', component_property='figure'),
    Output(component_id='ctf_png', component_property='figure')],
    [Input(component_id='mic_star', component_property='value'),
    Input(component_id='mic_scatter2D_graph', component_property='clickData')]
)
def load_mic_ctf(ctfstar_path, clickdata):

    # If there is no job file selected
    if ctfstar_path is None:
        mic_png = empty_graph("No micrograph data loaded")
        ctf_png = empty_graph("No CTF data loaded")
    
    # If no image is selecetd
    elif clickdata is None:
        mic_png = empty_graph("No micrograph selected")
        ctf_png = empty_graph("No CTF selected")

    else:
        
        # Get image index
        point_number = clickdata['points'][0]['pointNumber']
        
        ctf_df = starfile.read(ctfstar_path)['micrographs']
        # ctf_df.index += 1

        # Get image path
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

    return (mic_png, ctf_png)

## Callback Particles
@app.callback(
    [Output(component_id='ptcl_dropdown_x', component_property='options'),
     Output(component_id='ptcl_dropdown_y', component_property='options'),
     Output(component_id='ptcl_dropdown_class', component_property='options'),
     Output(component_id='ptcl_scatter2D_graph', component_property='figure'),
     Output(component_id='selected_particles', component_property='children')],
    [Input(component_id='ptcl_star', component_property='value'),
     Input(component_id='ptcl_dropdown_x', component_property='value'),
     Input(component_id='ptcl_dropdown_y', component_property='value'),
     Input(component_id='ptcl_dropdown_class', component_property='value'),
     State(component_id='ptcl_scatter2D_graph', component_property='selectedData'),
     State(component_id='basename_export_ptcl', component_property='value'),
     Input(component_id='export_particles_button', component_property='n_clicks'),
     ]
)

def load_df_and_graphs(ptcl_star, ptcl_dd_x, ptcl_dd_y, ptcl_dd_class,selectedPtclData,basename_export_ptcl,exportPtcl_button_press):

### Particles

    # Importing particles starfile data

    try:
        ptcl_df = starfile.read(ptcl_star)['particles']
        ptcl_df_optics = starfile.read(ptcl_star)['optics']
    except:
        print('No starfile selected')
        raise PreventUpdate

    # Exclude name labels for dropdowns
    dropdown_labels = [label for label in list(ptcl_df) if 'Name' not in label]
    ptcl_dropdown_x = dropdown_labels
    ptcl_dropdown_y = dropdown_labels
    ptcl_dropdown_class = dropdown_labels

    # Duplicating df info
    ptcl_dff = ptcl_df.copy()

    # Color definitions
    color1 = 'cornflowerblue'


    if ptcl_dd_class == None: ptcl_dd_class = 'cornflowerblue'
    else: ptcl_dd_class = ptcl_dff[ptcl_dd_class].astype(float)
    
    # Particles plot using WebGL given probably large dataframes
    ptcl_scatter2D = plot_scattergl(ptcl_dff, ptcl_dd_x, ptcl_dd_y, ptcl_dd_class, 'Viridis', 'cornflowerblue')
    
    # Parsing info from manual on-plot selection
    selected_particle_indices = []
    NOTselected_particle_indices = []
    if isinstance(selectedPtclData, dict):
        for i in selectedPtclData['points']:
            selected_particle_indices.append(int(i['pointIndex']))
        NOTselected_particles = ptcl_dff.loc[ptcl_dff.index.difference(selected_particle_indices)]
        NOTselected_particle_indices = list(NOTselected_particles.index.values)

    # Output definitions

    selectionPtcl_output = 'You\'ve selected '+str(len(selected_particle_indices))+' particles with indices: '+str(selected_particle_indices)

    outfile_ptcl_YES = str(basename_export_ptcl + '_selected_particles.star')
    outfile_ptcl_NO = str(basename_export_ptcl + '_not_selected_particles.star')

    exportPtcl_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'export_particles_button' in exportPtcl_button_press_changed:
        dict_ptcl_output_YES = {'optics' : ptcl_df_optics , 'particles' : ptcl_df.iloc[selected_particle_indices]}
        dict_ptcl_output_NO = {'optics' : ptcl_df_optics , 'particles' : ptcl_df.iloc[NOTselected_particle_indices]}
        starfile.write(dict_ptcl_output_YES, outfile_ptcl_YES, overwrite=True)
        starfile.write(dict_ptcl_output_NO, outfile_ptcl_NO, overwrite=True)
        print('Exported selected particles as '+ outfile_ptcl_YES + ' and not selected particles as ' + outfile_ptcl_NO)

    ### RETURN

    return ([ptcl_dropdown_x, ptcl_dropdown_y, ptcl_dropdown_class, ptcl_scatter2D, selectionPtcl_output])


## Callback 2D Classification
@app.callback(
    [Output(component_id='C2Dfollow_msg', component_property='value'),
     Output(component_id='C2Dfollow_graph', component_property='figure'),
     Output(component_id='C2Dclassnumber_graph', component_property='figure'),
     Output(component_id='C2Dfollow_dropdown_y', component_property='options')],
    [Input(component_id='C2Djob2follow', component_property='value'),
     Input(component_id='C2Dfollow_dropdown_y', component_property='value'),
     Input(component_id='C2Ddisplay_last_ite', component_property='n_clicks'),
     ]
)

def load_df_and_graphs(C2Djob2follow, C2Dfollow_dd_y, C2Ddisplay_last_ite_button_press):

### Follow 2D

    job = C2Djob2follow
    follow_dd_y = C2Dfollow_dd_y

    if 'Class2D' in str(job):

        C2Dfollow_msg = 'Following Class2D job: '+str(job)

        # Get a list of files matching regular expression "run_it*_optimiser.star"
        all_opt = glob.glob(os.path.join(job+'run_it*_optimiser.star'))
        all_opt.sort()

        stars_opt = []

        # Read each file and get it's information
        for filename in all_opt:
            optimiser_df = starfile.read(filename)
            stars_opt.append(optimiser_df)

        # Build dataframe
        follow_opt_df = pd.concat(stars_opt, axis=0, ignore_index=True)
        C2Dfollow_dd_y_list = list(follow_opt_df)

        # Get a list of files matching regular expression "run_it*_model.star"
        all_model = glob.glob(os.path.join(job+'run_it*_model.star'))
        all_model.sort()

        stars_classnumber = []

        # Read each file and get it's information
        for filename in all_model:
            classnumber_df = starfile.read(filename)['model_classes']
            classnumber_df = classnumber_df['rlnClassDistribution']
            stars_classnumber.append(classnumber_df)

        number_of_files = len(follow_opt_df[follow_dd_y])

        # Get the number of iterations, which depends on the algorithm used
        job_df = starfile.read(os.path.join(job, "job.star"))["joboptions_values"]
        if job_df[job_df["rlnJobOptionVariable"] == "do_em"]["rlnJobOptionValue"].item() == "Yes":
            number_of_iterations = job_df[job_df["rlnJobOptionVariable"] == "nr_iter_em"]["rlnJobOptionValue"].item()
        else:
            number_of_iterations = job_df[job_df["rlnJobOptionVariable"] == "nr_iter_grad"]["rlnJobOptionValue"].item() 
        number_of_iterations = int(number_of_iterations) 

        # Adjust dataframe for data display
        follow_classnumber_df = pd.concat(stars_classnumber, axis=1, ignore_index=True)
        follow_classnumber_df.index = follow_classnumber_df.index + 1
        follow_classnumber_df = follow_classnumber_df.T
        follow_classnumber_df.index = list(range(0, number_of_iterations + 1, (number_of_iterations//(number_of_files-1))))
        
    else:
        raise dash.exceptions.PreventUpdate

    # Plot parameters over iterations

    C2Dfollow_graph = plot_line(follow_opt_df[1:], list(range(number_of_iterations//(number_of_files-1), number_of_iterations + 1, (number_of_iterations//(number_of_files-1)))), follow_dd_y)
    C2Dfollow_graph.update_layout(title_text='Convergence', title_x=0.5, xaxis_title="Iteration",yaxis_title=C2Dfollow_dd_y)

    # Plot class distribution over iterations
    C2Dclassnumber_graph = plot_area(follow_classnumber_df)
    C2Dclassnumber_graph.update_layout(title_text='Class distribution', title_x=0.5,xaxis_title="Iteration",yaxis_title="Class proportion")

    print(C2Dfollow_msg)

    # Display last ite
    C2Ddisplay_last_ite_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_last_ite' in C2Ddisplay_last_ite_button_press_changed:
        print('displaying ' + all_opt[-1])
        os.system(str('`which relion_display` --i '+ all_opt[-1] +' --gui '))

    ### RETURN

    return ([C2Dfollow_msg, C2Dfollow_graph, C2Dclassnumber_graph, C2Dfollow_dd_y_list])


## Callback 3D Classification
@app.callback(
    [Output(component_id='follow_msg', component_property='value'),
     Output(component_id='follow_graph', component_property='figure'),
     Output(component_id='follow_dropdown_y', component_property='options'),
     Output(component_id='classnumber_graph', component_property='figure'),
     Output(component_id='follow_model', component_property='figure'),
     Output(component_id='follow_model_dropdown_y', component_property='options'),
     Output(component_id='angdist_per_class', component_property='figure'),
     Output(component_id='angdist_per_class', component_property='style')],
    [Input(component_id='job2follow', component_property='value'),
     Input(component_id='follow_dropdown_y', component_property='value'),
     Input(component_id='follow_model_dropdown_y', component_property='value'),
     Input(component_id='display_last_ite', component_property='n_clicks'),
     Input(component_id='display_chimera_last_ite', component_property='n_clicks')
    ]
)

def load_df_and_graphs(job2follow, follow_dd_y, follow_model_dd_y, display_last_ite_button_press, display_last_ite_chimera_button_press):

### Follow 3D

    job = job2follow

    if 'Class3D' in str(job):

        follow_msg = 'Following Class3D job: '+str(job)

        # Get optimiser files' data
        all_opt = glob.glob(os.path.join(job+'run_it*_optimiser.star'))
        all_opt.sort()
        stars_opt = []
        for filename in all_opt:
            optimiser_df = starfile.read(filename)
            stars_opt.append(optimiser_df)
        follow_opt_df = pd.concat(stars_opt, axis=0, ignore_index=True)
        follow_opt_df_last_ite_model = follow_opt_df['rlnModelStarFile'].iloc[-1]
        follow_opt_df_last_ite_maps = str(follow_opt_df_last_ite_model[:-10]+'class*mrc')
        follow_dd_y_list = list(follow_opt_df)

        # Get model files' data
        all_model = glob.glob(os.path.join(job+'run_it*_model.star'))
        all_model.sort()
        stars_classnumber = []
        for filename in all_model:
            classnumber_df = starfile.read(filename)['model_classes']
            classnumber_df = classnumber_df['rlnClassDistribution']
            stars_classnumber.append(classnumber_df)
        follow_classnumber_df = pd.concat(stars_classnumber, axis=1, ignore_index=True)
        follow_classnumber_df.index += 1
        follow_classnumber_df = follow_classnumber_df.T

        # Get last iteration's data
        last_ite_model = str(all_model[-1])
        last_ite_model_df = starfile.read(last_ite_model)['model_classes']
        last_ite_number_of_models = starfile.read(last_ite_model)['model_general']['rlnNrClasses']
        last_ite_fsc_df = pd.DataFrame()
        for i in range(1, int(last_ite_number_of_models)+1):
            classtable=str('model_class_')+str(i)
            last_ite_fsc_df_tmp = starfile.read(last_ite_model)[classtable]
            last_ite_fsc_df_tmp = last_ite_fsc_df_tmp.assign(class_number=i)
            if 'rlnResolution' in last_ite_fsc_df:
                last_ite_fsc_df = pd.concat([last_ite_fsc_df, last_ite_fsc_df_tmp], axis=0)
            else:
                last_ite_fsc_df = last_ite_fsc_df_tmp
        last_ite_fsc_df_list = list(last_ite_fsc_df)
        print(last_ite_fsc_df)
        follow_opt_df_last_ite_model = last_ite_model_df['rlnReferenceImage']
        resolutions = list(last_ite_fsc_df['rlnResolution'])

        line_coloring = 'class_number'
        number_of_iterations = len(follow_opt_df[follow_dd_y])
        number_of_its_class3D = len(follow_classnumber_df)-1
        angdist_per_class = starfile.read(job+f'run_it{number_of_its_class3D:03d}_data.star')['particles'][['rlnClassNumber', 'rlnAngleRot', 'rlnAngleTilt']]
        sampling_angle_class = float(starfile.read(job+f'run_it{number_of_its_class3D:03d}_sampling.star')['sampling_general']['rlnPsiStep'])

    else:
        raise dash.exceptions.PreventUpdate

    # Convergence plot
    follow_graph = plot_line(follow_opt_df[1:], list(range(1, number_of_iterations)), follow_dd_y)
    follow_graph.update_layout(title_text='Convergence', title_x=0.5, xaxis_title="Iteration",yaxis_title=follow_dd_y)

    # Last iteration plot
    follow_model = plot_line_multi(last_ite_fsc_df, resolutions, follow_model_dd_y, line_coloring)
    follow_model.update_layout(title_text='Convergence', title_x=0.5, xaxis_title="1/Resolution",yaxis_title=follow_model_dd_y)
    if follow_model_dd_y == 'rlnGoldStandardFsc': follow_model.add_hline(y=0.143, line_dash='dash', line_color="grey")
    elif follow_model_dd_y == 'rlnSsnrMap': follow_model.add_hline(y=1, line_dash='dash', line_color="grey")
    follow_model.update_layout(title_text=f'Iteration {number_of_iterations:03d}', title_x=0.5)

    # Class distribution plot
    classnumber_graph = plot_area(follow_classnumber_df)
    classnumber_graph.update_layout(title_text='Class distribution', title_x=0.5,xaxis_title="Iteration",yaxis_title="Class proportion")
    
    # Angular distributions plot
    angdist_per_class_plot = plot_angdist(angdist_per_class.sort_values('rlnClassNumber'), 'rlnAngleRot', 'rlnAngleTilt', int(360//(sampling_angle_class)), 'rlnClassNumber')
    angdist_per_class_plot.update_layout(
        xaxis = dict(tickmode= 'linear', dtick = 90),
        yaxis = dict(tickmode= 'linear', dtick = 90),
        title_text='Angular Distribution',
        title_x=0.5
    )
    classes = angdist_per_class["rlnClassNumber"].nunique()
    angdist_width = min(100, 100 * ((classes + 0.35) / 4))
    angdist_height = 100 * (1/4) * ((classes // 4) + (1 if classes % 4 != 0 else 0))
    angdist_per_class_plot_style = {'width' : f'{angdist_width}%', 'height' : f'{angdist_height}vw'}

    print(follow_msg)

    # Display last ite
    display_last_ite_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_last_ite' in display_last_ite_button_press_changed:
        print('displaying ' + all_opt[-1])
        os.system(str('`which relion_display` --i '+ all_opt[-1] + ' --gui '))

    display_last_ite_chimera_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_chimera_last_ite' in display_last_ite_chimera_button_press_changed:
        print('displaying ' + ' '.join(follow_opt_df_last_ite_model) + ' in chimera')
        os.system(str('`which chimera` '+ ' '.join(follow_opt_df_last_ite_model)))

    ### RETURN

    return ([follow_msg, follow_graph, follow_dd_y_list, classnumber_graph, follow_model, last_ite_fsc_df_list, angdist_per_class_plot, angdist_per_class_plot_style])

## Callback Refine 3D
@app.callback(
    [Output(component_id='ref_follow_msg', component_property='value'),
     Output(component_id='ref_follow_graph', component_property='figure'),
     Output(component_id='ref_follow_dropdown_y', component_property='options'),
     Output(component_id='ref_follow_model', component_property='figure'),
     Output(component_id='ref_follow_model_dropdown_y', component_property='options'),
     Output(component_id='ref_follow_angdist', component_property='figure')],
    [Input(component_id='ref_job2follow', component_property='value'),
     Input(component_id='ref_follow_dropdown_y', component_property='value'),
     Input(component_id='ref_follow_model_dropdown_y', component_property='value'),
     Input(component_id='ref_display_last_ite', component_property='n_clicks'),
     Input(component_id='ref_display_chimera_last_ite', component_property='n_clicks')
     ]
)

def load_df_and_graphs(ref_job2follow, ref_follow_dd_y,ref_follow_model_dd_y,ref_display_last_ite_button_press, ref_display_last_ite_chimera_button_press):

### Follow Refine 3D

    ref_job = ref_job2follow

    if 'Refine3D' in str(ref_job):

        ref_follow_msg = 'Following Refine3D job: '+str(ref_job)

        # Get model's data
        all_model = glob.glob(os.path.join(ref_job+'run_it*_half1_model.star'))
        all_model.sort()
        stars_model1 = []
        for filename in all_model:
            optimiser_df = starfile.read(filename)['model_general']
            stars_model1.append(optimiser_df)
        ref_follow_opt_df = pd.concat(stars_model1, axis=0, ignore_index=True)
        ref_follow_dd_y_list = list(ref_follow_opt_df)

        # Get last iteration's data
        last_ite_model = str(all_model[-1])
        last_ite_model_df = starfile.read(last_ite_model)['model_classes']
        last_ite_fsc_df = starfile.read(last_ite_model)['model_class_1']
        last_ite_fsc_df_list = list(last_ite_fsc_df)
        ref_follow_opt_df_last_ite_model = last_ite_model_df['rlnReferenceImage']
        ref_follow_opt_df_last_ite_maps = ref_follow_opt_df_last_ite_model
        resolutions = list(last_ite_fsc_df['rlnResolution'])

        number_of_iterations = len(ref_follow_opt_df[ref_follow_dd_y])-1

        angdist = starfile.read(ref_job+f'run_it{number_of_iterations:03d}_data.star')['particles'][['rlnAngleRot', 'rlnAngleTilt']]
        sampling_angle = float(starfile.read(ref_job+f'run_it{number_of_iterations:03d}_sampling.star')['sampling_general']['rlnPsiStep'])
        
    else:
        raise dash.exceptions.PreventUpdate

    # Convergence plot
    ref_follow_graph = plot_line(ref_follow_opt_df, range(number_of_iterations+1), ref_follow_dd_y)
    ref_follow_graph.update_layout(title_text='Convergence', title_x=0.5, xaxis_title='Iteration', yaxis_title=ref_follow_dd_y)

    # Last iteration plot
    ref_follow_model = plot_line(last_ite_fsc_df, resolutions, ref_follow_model_dd_y)
    if ref_follow_model_dd_y == 'rlnGoldStandardFsc': ref_follow_model.add_hline(y=0.143, line_dash='dash', line_color="grey")
    elif ref_follow_model_dd_y == 'rlnSsnrMap': ref_follow_model.add_hline(y=1, line_dash='dash', line_color="grey")
    ref_follow_model.update_layout(title_text=f'Iteration {number_of_iterations:03d}', title_x=0.5, xaxis_title='1/Resolution', yaxis_title = ref_follow_model_dd_y)

    # Angular distribution plot
    angdist_plot = plot_angdist(angdist, 'rlnAngleRot', 'rlnAngleTilt', int(360//(sampling_angle*2)), [1]*len(angdist['rlnAngleRot']))
    angdist_plot.update_layout(
        xaxis = dict(tickmode= 'linear', dtick = 90),
        yaxis = dict(tickmode= 'linear', dtick = 90),
        title_text = 'AngDist',
        title_x = 0.5
    )
    print(ref_follow_msg)

    # Display last ite
    ref_display_last_ite_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_last_ite' in ref_display_last_ite_button_press_changed:
        print('displaying ' + ref_follow_opt_df_last_ite_model[0])
        os.system(str('`which relion_display` --i '+ ref_follow_opt_df_last_ite_model[0] +' --gui '))

    ref_display_last_ite_chimera_button_press_changed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'display_chimera_last_ite' in ref_display_last_ite_chimera_button_press_changed:
        print('displaying ' + ref_follow_opt_df_last_ite_maps[0] + ' in chimera')
        os.system(str('`which chimera` '+ ref_follow_opt_df_last_ite_maps[0]))

    ### RETURN

    return ([ref_follow_msg, ref_follow_graph, ref_follow_dd_y_list, ref_follow_model, last_ite_fsc_df_list, angdist_plot])



### Dash app start

if __name__ == '__main__':
    app.run_server(debug=debug_mode, dev_tools_hot_reload = False, use_reloader=True,
                   host=hostname, port=port_number)
