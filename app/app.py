import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import infer
import torch
import dash_daq as daq
import os
from glob import glob

model = infer.Model("saved")
source_img = None
style_codes = [None, None, None, None]
source_img_1 = None
style_code_1 = None

boundary_files = sorted(glob(os.path.join("svms/male", '*.npy'))) + sorted(glob(os.path.join("svms/female", '*.npy')))
boundaries = []
for i in boundary_files:
    boundaries.append(np.load(i))

def b64_to_pil(string):
    decoded = base64.b64decode(string.split(',')[-1])
    buffer = BytesIO(decoded)
    im = Image.open(buffer).convert("RGB")
    return im

app = dash.Dash(__name__)
"""
Layout of the app
"""
app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Reference Interpolation', children=[
            html.Div([dcc.Upload(
                id='upload-data1',
                children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '250px','height': '60px','lineHeight': '60px', 'borderWidth': '1px',
                                                                                        'borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center'})],
                                                                                        style = {'display':'inline-block', 'margin-left':'350px'}),
            html.Div([dcc.Upload(
                id='upload-data2',
                children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '250px','height': '60px','lineHeight': '60px', 'borderWidth': '1px',
                                                                                        'borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center'})],
                                                                                        style = {'display':'inline-block', 'margin-left':'20px'}),
            html.Div([dcc.Upload(
                id='upload-data3',
                children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '250px','height': '60px','lineHeight': '60px', 'borderWidth': '1px',
                                                                                        'borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center'})],
                                                                                        style = {'display':'inline-block', 'margin-left':'20px'}),
            html.Div([dcc.Upload(
                id='upload-data4',
                children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '250px','height': '60px','lineHeight': '60px', 'borderWidth': '1px',
                                                                                        'borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center'})],
                                                                                        style = {'display':'inline-block', 'margin-left':'20px'}),

            html.Div(id = 'ref-img1', style = {'margin-left':'375px', 'margin-top':'10px', 'display':'inline-block'}),
            html.Div(id = 'ref-img2', style = {'display':'inline-block', 'margin-left':'70px', 'margin-top':'10px'}),
            html.Div(id = 'ref-img3', style = {'display':'inline-block', 'margin-left':'70px', 'margin-top':'10px'}),
            html.Div(id = 'ref-img4', style = {'display':'inline-block', 'margin-left':'70px', 'margin-top':'10px'}),
            
            html.Br(),
            
            html.Div([daq.ToggleSwitch(
                id='y1',
                value=False,
                label=['Female','Male'],
            )], style={'display': 'inline-block', 'margin-left':'400px'}),

            html.Div([daq.ToggleSwitch(
                id='y2',
                value=False,
                label=['Female','Male'],
            )], style={'display': 'inline-block', 'margin-left':'120px'}),
            
            html.Div([daq.ToggleSwitch(
                id='y3',
                value=False,
                label=['Female','Male'],
            )], style={'display': 'inline-block', 'margin-left':'120px'}),
            
            html.Div([daq.ToggleSwitch(
                id='y4',
                value=False,
                label=['Female','Male'],
            )], style={'display': 'inline-block', 'margin-left':'120px'}),

            html.Div([dcc.Upload(
                id='upload-data5',
                children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '200px','height': '60px','lineHeight': '60px', 'borderWidth': '1px',
                                                                                        'borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center'})],
                                                                                        style = {'margin-left':'150px'}),
            
            html.Div(id = 'src_img', style = {'margin-left':'155px', 'margin-top':'10px', 'display':'inline-block'}),
            
            html.Div([dcc.Slider(
                        id='style-slider',
                        min=0,
                        max=300,
                        step=1,
                        value=100,
                        marks={
                        0: {'label': '1'},
                        100: {'label': '2'},
                        200: {'label': '3'},
                        300: {'label': '4'},
                        })],
                        style={'width':'50%','display': 'inline-block', 'margin-left':'75px'}),         

            html.Img(id="out_img", src = "//:0",style={'height':'200px', 'width':'200px'}),
        ]),
        dcc.Tab(label='Style Interpolation', children=[
            html.Div([dcc.Upload(
                id='upload-datastyle1',
                children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '250px','height': '60px','lineHeight': '60px', 'borderWidth': '1px',
                                                                                        'borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center'})],
                                                                                        style = {'display':'inline-block', 'margin-left':'350px'}),
            html.Div([dcc.Upload(
                id='upload-datastyle2',
                children=html.Div(['Drag and Drop or ',html.A('Select Files')]),style={'width': '250px','height': '60px','lineHeight': '60px', 'borderWidth': '1px',
                                                                                        'borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center'})],
                                                                                        style = {'display':'inline-block', 'margin-left':'150px'}),
            html.Br(),
            
            html.Div(id = 'styleinterp_src', style = {'margin-left':'325px', 'margin-top':'10px', 'display':'inline-block'}),
            html.Div(id = 'styleinterp_ref', style = {'display':'inline-block', 'margin-left':'100px', 'margin-top':'10px'}),
            html.Img(id="styleinterp_out", src = "//:0",style={'height':'300px', 'width':'300px', 'margin-left':'200px'}),

            html.Br(),
            html.Div([daq.ToggleSwitch(
                id='styleinterp_y1',
                value=False,
                label=['Female','Male'],
            )], style={'display': 'inline-block', 'margin-left':'800px'}),
            
            html.Br(),
            html.Div([dcc.Slider(
                        id='styleinterp-slider1',
                        min=-10,
                        max=10,
                        step=0.5,
                        value=0,
                        marks={
                        -10: {'label': 'Male:Hair'},
                        10: {'label': 'Male:Bald'},
                        }
                        )],
                        style={'width':'50%','display': 'inline-block', 'margin-top':'10px', 'margin-left':'400px'}), 
            html.Div([dcc.Slider(
                        id='styleinterp-slider2',
                        min=-10,
                        max=10,
                        step=0.5,
                        value=0,
                        marks={
                        -10: {'label': 'Male:Less Bangs'},
                        10: {'label': 'Male:More Bangs'},
                        }
                        )],
                        style={'width':'50%','display': 'inline-block', 'margin-top':'10px', 'margin-left':'400px'}), 
            html.Div([dcc.Slider(
                        id='styleinterp-slider3',
                        min=-10,
                        max=10,
                        step=0.5,
                        value=0,
                        marks={
                        -10: {'label': 'Male:Less Black Hair'},
                        10: {'label': 'Male:More Black Hair'},
                        }
                        )],
                        style={'width':'50%','display': 'inline-block', 'margin-top':'10px', 'margin-left':'400px'}), 
            html.Div([dcc.Slider(
                        id='styleinterp-slider4',
                        min=-10,
                        max=10,
                        step=0.5,
                        value=0,
                        marks={
                        -10: {'label': 'Male:No Moustache'},
                        10: {'label': 'Male:Moustache'},
                        }
                        )],
                        style={'width':'50%','display': 'inline-block', 'margin-top':'10px', 'margin-left':'400px'}), 
            html.Div([dcc.Slider(
                        id='styleinterp-slider5',
                        min=-10,
                        max=10,
                        step=0.5,
                        value=0,
                        marks={
                        -10: {'label': 'Male:Beard'},
                        10: {'label': 'Male:No Beard'},
                        }
                        )],
                        style={'width':'50%','display': 'inline-block', 'margin-top':'10px', 'margin-left':'400px'}), 
            html.Div([dcc.Slider(
                        id='styleinterp-slider6',
                        min=-10,
                        max=10,
                        step=0.5,
                        value=0,
                        marks={
                        -10: {'label': 'Male: older'},
                        10: {'label': 'Male:younger'},
                        }
                        )],
                        style={'width':'50%','display': 'inline-block', 'margin-top':'10px', 'margin-left':'400px'}), 
            html.Div([dcc.Slider(
                        id='styleinterp-slider7',
                        min=-10,
                        max=10,
                        step=0.5,
                        value=0,
                        marks={
                        -10: {'label': 'Female:Less Black Hair'},
                        10: {'label': 'Female: More Black Hair'},
                        }
                        )],
                        style={'width':'50%','display': 'inline-block', 'margin-top':'10px', 'margin-left':'400px'}), 
            html.Div([dcc.Slider(
                        id='styleinterp-slider8',
                        min=-10,
                        max=10,
                        step=0.5,
                        value=0,
                        marks={
                        -10: {'label': 'Female:Curly Hair'},
                        10: {'label': 'Female:Straight Hair'},
                        }
                        )],
                        style={'width':'50%','display': 'inline-block', 'margin-top':'30px', 'margin-left':'400px'}), 
        ]),
    ])
])

"""
callback to populate uploaded src, and reference images. 
Also populates the global variable source image and style codes.
called with either new images are uploaded or the label toggles are triggered
"""
@app.callback([
    Output('ref-img1', 'children'), Output('ref-img2', 'children'),
    Output('ref-img3', 'children'), Output('ref-img4', 'children'),
    Output('src_img', 'children'),
],
[
    Input('upload-data1', 'contents'), Input('upload-data2', 'contents'), 
    Input('upload-data3', 'contents'), Input('upload-data4', 'contents'),
    Input('upload-data5', 'contents'), 
    Input('y1', 'value'), Input('y2', 'value'),
    Input('y3', 'value'), Input('y4', 'value'),
])
def load_imgs(ref_img1, ref_img2, ref_img3, ref_img4, src_img, y1, y2, y3, y4):
    global style_codes, source_img

    a = html.Img(src = ref_img1,style={'height':'200px', 'width':'200px'})
    b = html.Img(src = ref_img2,style={'height':'200px', 'width':'200px'})
    c = html.Img(src = ref_img3,style={'height':'200px', 'width':'200px'})
    d = html.Img(src = ref_img4,style={'height':'200px', 'width':'200px'})
    e = html.Img(src = src_img,style={'height':'200px', 'width':'200px'})

    if ref_img1 is not None:
        ref1 = b64_to_pil(ref_img1)
        ref1 = infer.TRANSFORM(ref1)
        y = np.array([0]) if y1 is False else np.array([1])
        style_codes[0] = model.get_style(ref1, y)
        
    if ref_img2 is not None:
        ref2 = b64_to_pil(ref_img2)
        ref2 = infer.TRANSFORM(ref2)
        y = np.array([0]) if y2 is False else np.array([1])
        style_codes[1] = model.get_style(ref2, y)
        
    if ref_img3 is not None:
        ref3 = b64_to_pil(ref_img3)
        ref3 = infer.TRANSFORM(ref3)
        y = np.array([0]) if y3 is False else np.array([1])
        style_codes[2] = model.get_style(ref3, y)
        
    if ref_img4 is not None:
        ref4 = b64_to_pil(ref_img4)
        ref4 = infer.TRANSFORM(ref4)
        y = np.array([0]) if y4 is False else np.array([1])
        style_codes[3] = model.get_style(ref4, y)
    
    if src_img is not None:
        src_img = b64_to_pil(src_img)
        source_img = infer.TRANSFORM(src_img)

    return a,b,c,d,e

"""
interpolating between reference images and generating output
"""
@app.callback(
    Output('out_img', 'src'), 
    Input('style-slider', 'value')
)
def translate_with_ref(value):
    global style_codes, source_img
    indx, interpolate = int(value//100), int(value%100)
    src = src='//:0'
    if source_img is not None:
        if interpolate!=0:
            style_ref = torch.lerp(style_codes[indx], style_codes[indx+1], interpolate/100)
            out_img = model.apply_style(source_img, style_ref)
        else:
            style_ref = style_codes[indx]
            out_img = model.apply_style(source_img, style_ref)
        infer.save_image(out_img, 1, "out.png")
        out = base64.b64encode(open("out.png", 'rb').read()).decode('ascii')
        src = src='data:image/png;base64,{}'.format(out)
    return src

"""
load images for second tab for style manipulation.
"""
@app.callback([
    Output('styleinterp_src', 'children'), Output('styleinterp_ref', 'children'),
],
[
    Input('upload-datastyle1', 'contents'), Input('upload-datastyle2', 'contents'), 
    Input('styleinterp_y1', 'value')
])
def load_imgs(src_img, ref_img, y1):
    global source_img_1, style_code_1

    a = html.Img(src = src_img,style={'height':'300px', 'width':'300px'})
    b = html.Img(src = ref_img,style={'height':'300px', 'width':'300px'})

    if ref_img is not None:
        ref = b64_to_pil(ref_img)
        ref = infer.TRANSFORM(ref)
        y = np.array([0]) if y1 is False else np.array([1])
        style_code_1 = model.get_style(ref, y)
        
    if src_img is not None:
        src_img = b64_to_pil(src_img)
        source_img_1 = infer.TRANSFORM(src_img)

    return a,b

"""
linearly interpolating style code based on svm boundaries and hence manipulate particular boundary
"""
@app.callback(
    Output('styleinterp_out', 'src'), 
    [
        Input('styleinterp-slider1', 'value'),Input('styleinterp-slider2', 'value'),
        Input('styleinterp-slider3', 'value'),Input('styleinterp-slider4', 'value'),
        Input('styleinterp-slider5', 'value'),Input('styleinterp-slider6', 'value'),
        Input('styleinterp-slider7', 'value'),Input('styleinterp-slider8', 'value')
    ]
)
def linear_interpolate(value1,value2,value3,value4,value5,value6,value7,value8):
    global style_code_1, source_img_1, boundaries
    ctx = dash.callback_context
    values = [value1, value2, value3, value4, value5, value6, value7, value8]
    if not ctx.triggered:
        pass
    else:
        id = ctx.triggered[0]["prop_id"].split(".")[0]
        boundary = boundaries[int(id[-1])-1]
        value = values[int(id[-1])-1]
    src = src='//:0'
    if source_img_1 is not None:
        value = value - style_code_1.reshape(1,-1).numpy().dot(boundary.T)
        value = value.reshape(-1, 1).astype(np.float32)
        style_code_1 = style_code_1 + value * boundary
        style_ref = style_code_1
        out_img = model.apply_style(source_img_1, style_ref)
        infer.save_image(out_img, 1, "out.png")
        out = base64.b64encode(open("out.png", 'rb').read()).decode('ascii')
        src = src='data:image/png;base64,{}'.format(out)
    return src

if __name__ == '__main__':
    app.run_server(debug=True)