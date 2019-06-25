# viz.py

from config import *


# https://plot.ly/python/apache-spark/
# https://plot.ly/python/offline/
import plotly.offline as pyo
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff

init_notebook_mode(connected=True)


def grouped_bar_plot(df_list, title_list, group_col):
    '''
    https://plot.ly/python/bar-charts/#grouped-bar-chart
    '''
    bar_data = []

    for df, title in zip(df_list, title_list):
        df_by = df.groupBy(group_col).agg(F.count('*').alias('count'))

        #print('Number of items for each group in {title} dataset:'.format(title=title))
        #df_by.show()

        df_by = df_by.collect()

        trace = go.Bar(x = [i[group_col] for i in df_by], y = [i['count'] for i in df_by], name=title)

        bar_data.append(trace)

    layout = go.Layout(barmode='group')

    fig = go.Figure(data=bar_data, layout=layout)
    iplot(fig, filename='grouped-bar')


def boxplots(data, columns):
    '''
    https://plot.ly/python/box-plots/
    https://dataplatform.cloud.ibm.com/exchange/public/entry/view/d80de77f784fed7915c14353512ef14d
    '''
    data_pd = data.select(columns).toPandas()

    traces = []

    for colname in columns:
        traces.append(go.Box(y = data_pd[colname], name = colname))
    
    return iplot(traces)


def dist_plots(data, columns, show_hist=True):
    '''
    - https://plot.ly/python/distplot/
    - https://en.wikipedia.org/wiki/Kernel_density_estimation
    '''
    hist_data = []
    colors = ['#333F44', '#37AA9C', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4']

    for colname in columns:
        df = data.select(colname).toPandas()[colname]
        hist_data.append(df)

    fig = ff.create_distplot(hist_data, columns, show_hist=show_hist, show_rug=False)
    fig['layout'].update(title='KDE curve plots')

    iplot(fig, filename='Kernel density estimation curve plots')


def line_plot(x, y, title, x_title, y_title, x_range=None, y_range=None):
    '''
    https://plot.ly/python/line-charts/#simple-line-plot
    '''
    xaxis = dict(title = x_title, range=x_range)
    yaxis = dict(title = y_title, range=y_range)
    layout = dict(title = title, xaxis = xaxis, yaxis = yaxis)
    data = [go.Scatter(x = x, y = y)]
    fig = dict(data=data, layout=layout)
    iplot(fig)
