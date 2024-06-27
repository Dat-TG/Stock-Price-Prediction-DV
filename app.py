#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            Phật phù hộ, không bao giờ BUG
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
app.title = "Stock Price Analysis Dashboard - LCD"
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))



df_nse = pd.read_csv("./csvdata/BID.csv")

data=df_nse.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset.loc[i, "Date"] = data.iloc[i]['Date']  # Correct assignment using iloc
    new_dataset.loc[i, "Close"] = data.iloc[i]["Price"]

new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

final_dataset=new_dataset.values

train_size = int(len(final_dataset) * 0.8)
train_data = final_dataset[:train_size, :]
valid_data = final_dataset[train_size:, :]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))


model = load_model("saved_model.keras")

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

train_data=new_dataset[:train_size]
valid_data=new_dataset[train_size:]
valid_data['Predictions']=predicted_closing_price



df= pd.read_csv("./csvdata/full_data_processed.csv")

# Drop duplicate rows based on 'Bank' and 'Stock symbol' columns
unique_df = df.drop_duplicates(subset=['Bank', 'Stock symbol'])

# Create the stock_list from the unique rows
stock_list = [{'label': bank, 'value': symbol} for bank, symbol in zip(unique_df['Bank'], unique_df['Stock symbol'])]

# Create the dropdown dictionary for easy lookup
dropdown_dict = {row['Stock symbol']: row['Bank'] for index, row in unique_df.iterrows()}

app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.P([html.B("Developed by: "), "Group 23"]),
        html.P("20120454 - Lê Công Đắt"),
        html.P("21120279 - Lê Trần Minh Khuê"),
        html.P("21120290 - Hoàng Trung Nam"),
        html.P("21120296 - Lê Trần Như Ngọc"),
        html.P("21120533 - Lê Thị Minh Phương"),
    ], style={"border": "1px solid #000", "padding": "10px", "marginBottom": "50px", "textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Predict Stock Price', children=[
            html.Div([
                dcc.Dropdown(id='stock-dropdown',
                             options=stock_list,
                             value='BID',
                             style={"width": "50%", "margin": "16px auto"}),
                html.H2("Actual vs Predicted Closing Prices", style={"textAlign": "center"}),
                dcc.Graph(
                    id="BTC Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Close"],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue')
                            ),
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Predictions"],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red')
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual vs Predicted Closing Prices for BIDV',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'},
                            legend=dict(x=0, y=1, traceorder='normal'),
                            hovermode='x'
                        )
                    }
                ),
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data AAPL",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Close"],
                                mode='markers',
                                name='Actual'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Actual Closing Price for BIDV',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data AAPL",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Predictions"],
                                mode='markers',
                                name='Predicted'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter plot of Predicted Closing Price for BIDV',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Comparison between stocks', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=stock_list, 
                             multi=True,value=[stock_list[0]['value']],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=stock_list, 
                             multi=True,value=[stock_list[0]['value']],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])


@app.callback(
    [Output('BTC Data', 'figure'),
     Output('Actual Data AAPL', 'figure'),
     Output('Predicted Data AAPL', 'figure')],
    [Input('stock-dropdown', 'value')]
)
def update_graph(selected_stock):
    # Ensure selected_stock is not None
    if selected_stock is None:
        # Return empty figures if selected_stock is None
        return {}, {}, {}

    df_stock = df[df['Stock symbol'] == selected_stock]

    # Ensure data is sorted by index if necessary
    data = df_stock.sort_index(ascending=True, axis=0)

    new_dataset = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_dataset.loc[i, "Date"] = data.iloc[i]['Date']  # Correct assignment using iloc
        new_dataset.loc[i, "Close"] = data.iloc[i]["Price"]

    new_dataset.set_index('Date', inplace=True)

    final_dataset = new_dataset.values

    train_size = int(len(final_dataset) * 0.8)
    train_data = final_dataset[:train_size, :]
    valid_data = final_dataset[train_size:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    model = load_model("saved_model.keras")

    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_closing_price = model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

    train_data = new_dataset[:train_size]
    valid_data = new_dataset[train_size:]
    valid_data['Predictions'] = predicted_closing_price

    actual_trace = go.Scatter(
        x=valid_data.index,
        y=valid_data["Close"],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue')
    )

    predicted_trace = go.Scatter(
        x=valid_data.index,
        y=valid_data["Predictions"],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red')
    )

    actual_scatter_trace = go.Scatter(
        x=valid_data.index,
        y=valid_data["Close"],
        mode='markers',
        name='Actual'
    )

    predicted_scatter_trace = go.Scatter(
        x=valid_data.index,
        y=valid_data["Predictions"],
        mode='markers',
        name='Predicted'
    )

    stock_name = dropdown_dict[selected_stock]

    figure1 = {
        "data": [actual_trace, predicted_trace],
        "layout": go.Layout(
            title=f'Actual vs Predicted Closing Prices for {stock_name}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'},
            legend=dict(x=0, y=1, traceorder='normal'),
            hovermode='x'
        )
    }

    figure2 = {
        "data": [actual_scatter_trace],
        "layout": go.Layout(
            title=f'Scatter plot of Actual Closing Price for {stock_name}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'}
        )
    }

    figure3 = {
        "data": [predicted_scatter_trace],
        "layout": go.Layout(
            title=f'Scatter plot of Predicted Closing Price for {stock_name}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'}
        )
    }

    return figure1, figure2, figure3






@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = dropdown_dict
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock symbol"] == stock]["Date"],
                     y=df[df["Stock symbol"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock symbol"] == stock]["Date"],
                     y=df[df["Stock symbol"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = dropdown_dict
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock symbol"] == stock]["Date"],
                     y=df[df["Stock symbol"] == stock]["Vol."],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)