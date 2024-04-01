import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
list_of_stocks = ['BAJFINANCE.csv', 'BPCL.NS.csv', 'HDB.csv', 'INFY.csv', 'KOTAKBANK.NS.csv',
                  'TATASTEEL.NS.csv','TCS.csv', 'TTNP.csv', 'WIT.csv']
stocks = {i:pd.read_csv(j) for i, j in enumerate(list_of_stocks)}
for i in stocks:
    stocks[i]=stocks[i].drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])

def R(price_col):
    first_row = [0]
    for i in range(1, len(price_col)):
        first_row.append(((price_col[i]-price_col[i-1])/(price_col[i-1]))*100)
    return np.array(first_row).round(2)
for i in stocks:
    stocks[i][f'R({i})'] = R(stocks[i].iloc[:, 1])


class Portfolio:
    def __init__(self, *stocks):
        self.stocks = {i:j for i, j in enumerate(stocks)}
            
#         within stocks we expect an array of df
        
        self.no_of_stocks = len(self.stocks)
        
    def getAllWeights(self):
        weight_options = range(0, 101, 5)
        combinations = [combo for combo in itertools.product(weight_options, repeat=len(self.stocks)) if sum(combo) == 100]
        return combinations
    
    
    
    def getPortReturn(self, *weights):
        avg = []
        for i in self.stocks:
            avg.append(self.stocks[i].iloc[:, -1].mean())
        avg = np.array(avg)
        weights = np.array(weights)[0]
        return np.dot(avg, weights/100).round(5)
    
    
    
    
    def getPortStd(self, *weights):
        stocks_returns = []
        for i in self.stocks:
            stocks_returns.append(self.stocks[i].iloc[:, -1])
        stocks_returns = np.array(stocks_returns)
        df = pd.DataFrame(stocks_returns.T)
        l = []
        weights = np.array(weights)[0]
        for i in df.index:
            l.append(np.dot(df.iloc[i, :],np.array(weights)/100))
        df['Portfolio(R)'] = l
        return df.iloc[:, -1].std()

import plotly.express as px
import plotly.offline as offline
def PlotlyCode(port1):
    x = []
    y = []
    for i in port1.getAllWeights():
        x.append(port1.getPortStd(i))
        y.append(port1.getPortReturn(i))
    fig = px.scatter(x=x, y = y, title="Efficient Frontier", labels={'x':'Risk', 'y':'Return'}, color=x)
    fig.update_layout(coloraxis_colorbar=dict(title='Risk'), title={'x':0.5, 'xanchor':'center'})
    html_string = offline.plot(fig, include_plotlyjs=False, output_type='div')
    return html_string
# fig.show()