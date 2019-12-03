#%%
## ~~~~~~~~~~ PART 0: DEFINITIONS ~~~~~~~~~~ ##
import pandas as pd
# from numpy import mean
# from json import load
import plotly.graph_objects as go
import os

from dash import Dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import numpy as np

#%%
## ~~~~~~~~~~ PART 1: DATA ~~~~~~~~~~ ##
os.chdir("/Users/tess/Desktop/Desktop/SCSML/assignments/final project/ecommerce-nlp/src")
df_og = pd.read_csv("../data/Womens Clothing E-Commerce Reviews.csv")
df_og.columns = ["clothing_id","age","title","review","rating", \
            "recomend","pos_fb_ct","division","department","class"]

term_details = pd.read_csv("../out/term_cat_details.csv")

#%%
## ~~~~~~~~~~ PART 2: MAP ~~~~~~~~~~ ##
app = Dash("E-Commerce Product Review Dashboard")
server = app.server

colors = {
    'accent1': '#1D3236',
    'accent2': '#1D7874',
    'light': '#FCFCFC',
    'medium': '#757575',
    'dark': '#222222'
}

def init_term_chart():
    fig = go.Figure()
    return fig

def get_term_chart(div,dept,cls,rec):
    running = term_details.copy()
    filters = {
        "division":div,
        "department":dept,
        "class":cls,
        "recomend":rec
    }
    for lbl,filt in filters.items():
        if filt is not None and len(filt)>0:
            running = running[[i in filt for i in running[lbl]]]
    vals_all = running.groupby("term").agg({"imp":"max","class":"count","cat":"max","rating":np.mean}).reset_index()
    vals_all["display_text"] = vals_all.apply(lambda x: "<b>Term</b>: {}</br><b>Category</b>: {}</br><b>Avg. Rating</b>: {}/5</br><b>Importance</b>: {}".format(x["term"],x["cat"],x["rating"],x["imp"]), axis=1)

    fig = go.Figure()
    for v in vals_all['cat'].drop_duplicates():
        vals = vals_all[vals_all["cat"]==v]
        fig.add_trace(go.Scatter(
            x= vals["imp"].astype(float),
            y= vals["class"].astype(float),
            name= vals["cat"].iloc[0],
            text= list(vals["display_text"]),
            marker_size= (vals["rating"]*100).astype(float)
        ))
    
    fig.update_traces(mode='markers', marker=dict(sizemode='area',line_width=2))
    
    fig.update_layout(
        title='Importance of Terms for Recommendations v. Ratings',
        xaxis=dict(
            title='Importance of term in model',
            gridwidth=2,
        ),
        yaxis=dict(
            title='Term frequency for selected products',
            gridwidth=2,
        ),
        font= {'color': colors['medium']}
    )
    return [fig]
    # prior: make table with top ~30 terms (onehot), filter columns, rating, category
    #        make dict of importance {"term":<scaled_value>}
    # size by average rating associated with term?
    # y by frequency -- filter down then count
    # x by importance = sorted([100*i for i in model.feature_importances_ if i!=0])[-30:]
    # colour by category -- apply term model
    # importance
    # map to labels
    # size by frequency within

def get_rating_recommendation_data(div=None,dept=None,cls=None,rec=None):
    running = df_og.copy()
    filters = {
        "division":div,
        "department":dept,
        "class":cls,
        "recomend":rec
    }
    for lbl,filt in filters.items():
        if filt is not None and len(filt)>0:
            running = running[[i in filt for i in running[lbl]]]
    vals = running.groupby(["rating","recomend"]).agg({"review":"count"}).reset_index()
    rec_dct = vals[vals['recomend']==1].set_index("rating")['review'].to_dict()
    nrec_dct = vals[vals['recomend']==0].set_index("rating")['review'].to_dict()
    ratings = [i for i in range(1,6)]
    fig = go.Figure(data=[
        go.Bar(name='Recommended', x=ratings, y=[rec_dct[i] if i in rec_dct.keys() else 0 for i in ratings]),
        go.Bar(name='Not Recommended', x=ratings, y=[nrec_dct[i]  if i in nrec_dct.keys() else 0 for i in ratings])
    ])
    fig.update_layout(title='Recommendations and Ratings',
                        barmode='stack',
                        clickmode='event+select',
                        xaxis={"title":"Rating"},
                        yaxis={"title":"Number of Reviews",
                            "rangemode":"tozero"},
                        plot_bgcolor=colors['light'],
                paper_bgcolor=colors['light'],
                font= {'color': colors['medium']})
    return [fig]

def init_rating_recommendation_data():
    vals = df_og.groupby(["rating","recomend"]).agg({"review":"count"}).reset_index()
    rec_dct = vals[vals['recomend']==1].set_index("rating")['review'].to_dict()
    nrec_dct = vals[vals['recomend']==0].set_index("rating")['review'].to_dict()
    ratings = [i for i in range(1,6)]
    fig = go.Figure(data=[
        go.Bar(name='Recommended', x=ratings, y=[rec_dct[i] if i in rec_dct.keys() else 0 for i in ratings]),
        go.Bar(name='Not Recommended', x=ratings, y=[nrec_dct[i]  if i in nrec_dct.keys() else 0 for i in ratings])
    ])
    fig.update_layout(title='Recommendations and Ratings',
                        barmode='stack',
                        clickmode='event+select',
                        xaxis={"title":"Rating"},
                        yaxis={"title":"Number of Reviews",
                            "rangemode":"tozero"},
                        plot_bgcolor=colors['light'],
                paper_bgcolor=colors['light'],
                font= {'color': colors['medium']})
    return fig

app.layout = html.Div([
    html.Div([
        html.H3("Filters",style={"padding-left":"1vw"}),
        html.Label('Product Division',style={"padding-left":"1vw"}),
        dcc.Dropdown(
            id='division-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df_og.division.drop_duplicates().dropna())],
            clearable=True,
            searchable=True,
            placeholder="Division",
            multi=True,
            style=dict(
                    width='96%',
                    horizonalAlign="middle",
                    color=colors['dark'],
                    padding="2%"
            )
        ),

        html.Label('Product Department',style={"padding-left":"1vw"}),
        dcc.Dropdown(
            id='department-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df_og.department.drop_duplicates().dropna())],
            clearable=True,
            searchable=True,
            placeholder="Department",
            multi=True,
            style=dict(
                    width='96%',
                    horizonalAlign="middle",
                    color=colors['dark'],
                    padding="2%"
            )
        ),

        html.Label('Product Class',style={"padding-left":"1vw"}),
        dcc.Dropdown(
            id='class-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df_og["class"].drop_duplicates().dropna())],
            clearable=True,
            searchable=True,
            multi=True,
            style=dict(
                    width='96%',
                    horizonalAlign="middle",
                    color=colors['dark'],
                    padding="2%"
            )
        ),

        html.Label('Recommendation',style={"padding-left":"1vw"}),
        dcc.Dropdown(
            id='recommend-dropdown',
            options=[
                {'label': 'Recommended', 'value': 1},
                {'label': 'Not Recommended', 'value': 0}
            ],
            clearable=True,
            searchable=True,
            multi=True,
            style=dict(
                    width='96%',
                    horizonalAlign="middle",
                    color=colors['dark'],
                    padding="2%"
            )
        ),

        # html.Div([html.Label('Rating'),
        # dcc.RangeSlider(
        #     id="rating-slider",
        #     marks={i: '{}'.format(i) for i in range(1,6)},
        #     min=1,
        #     max=5,
        #     value=[1,5],
        #     included=True
        # )],style=dict(
        #             width='80%',
        #             horizonalAlign="middle",
        #             color=colors['light']
        #     ))
    ], style = dict(
            width = '20%',
            display = "inline-block",
            backgroundColor=colors['dark'],
            color = colors['light']
        )
    ),
    html.Div([
        html.H1("E-Commerce Product Review Dashboard"),
        dcc.Graph(
            id='ratings-graph',
            figure=init_rating_recommendation_data()
        ),
        dcc.Graph(
            id='term-graph',
            figure=init_term_chart()
        )
    ], style = dict(
            width = '70%',
            display = "inline-block",
            backgroundColor=colors['light'],
            color = colors['dark'],
            verticalAlign="top",
            textAlign='center'
        ))
],
style = dict(
    backgroundColor=colors['dark'],
    color = colors['light'])
)

@app.callback(
    [Output('ratings-graph', 'figure')],
    [Input('division-dropdown', 'value'),
    Input('department-dropdown', 'value'),
    Input('class-dropdown', 'value'),
    Input('recommend-dropdown', 'value')])
def update_rating_recommendation_data(div,dept,cls,rec):
    return get_rating_recommendation_data(div,dept,cls,rec)

@app.callback(
    [Output('term-graph', 'figure')],
    [Input('division-dropdown', 'value'),
    Input('department-dropdown', 'value'),
    Input('class-dropdown', 'value'),
    Input('recommend-dropdown', 'value')])
def update_rating_recommendation_data(div,dept,cls,rec):
    return get_term_chart(div,dept,cls,rec)



@app.callback(
    [Output('department-dropdown', 'options'),
    Output('department-dropdown', 'value')],
    [Input('division-dropdown', 'value')])
def set_dept_options_values(div):
    if div is None or len(div) < 1:
        return [{'label': i, 'value': i} for i in sorted(df_og.department.drop_duplicates().dropna())], []
    return [{'label': i, 'value': i} for i in sorted(df_og[[i in div for i in df_og['division']]].department.drop_duplicates().dropna())], []


@app.callback(
    [Output('class-dropdown', 'options'),
    Output('class-dropdown', 'value')],
    [Input('division-dropdown', 'value'),
    Input('department-dropdown', 'value')])
def set_class_options_values(div,dept):
    if (div is None or len(div)<1) and (dept is None or len(dept) < 1):
        return [{'label': i, 'value': i} for i in sorted(df_og["class"].drop_duplicates().dropna())], []
    if dept is None or len(dept)>=1:
        return [{'label': i, 'value': i} for i in sorted(df_og[[i in dept for i in df_og['department']]]["class"].drop_duplicates().dropna())], []
    return [{'label': i, 'value': i} for i in sorted(df_og[[i in div for i in df_og['division']]]["class"].drop_duplicates().dropna())], []


#%%
if __name__ == '__main__':
    app.run_server(debug=True)
