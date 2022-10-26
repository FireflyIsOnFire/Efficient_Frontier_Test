import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import date
import numpy.random as npr
import matplotlib.pyplot as plt
from pylab import mpl
import scipy.optimize as sco
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus']=False

#此代码为手动调整电脑收集相应股票股价，并且通过concat合并到一张表中：
'''
begin=date(2020,1,1)
end=date(2020,12,31)
df1 = web.DataReader('BTC-USD','yahoo',begin,end)['Adj Close']
return_df1=np.log(df1/df1.shift(1))
df2 = web.DataReader('ETH-USD','yahoo',begin,end)['Adj Close']
return_df2=np.log(df2/df2.shift(1))
df3 = web.DataReader('USDT-USD','yahoo',begin,end)['Adj Close']
return_df3=np.log(df3/df3.shift(1))
df4 = web.DataReader('ADA-USD','yahoo',begin,end)['Adj Close']
return_df4=np.log(df4/df4.shift(1))
df5 = web.DataReader('XRP-USD','yahoo',begin,end)['Adj Close']
return_df5=np.log(df5/df5.shift(1))
df6 = web.DataReader('SOL-USD','yahoo',begin,end)['Adj Close']
return_df6=np.log(df6/df6.shift(1))


data = pd.concat([return_df1,return_df2,return_df3,return_df4,return_df5,return_df6],axis = 1)  # concat函数将这些结果都并在一个数列中
data.columns = ['BTC','ETF','USDT','ADA','XRP','SOL']
data.to_csv('gushi.csv',encoding = 'utf-8')  #encoding = 'utf-8' 不明白什么意思
'''


#我想测试一下，如果使用for循环单次完成所有信息收集是否也可搜集在总的csv中
'''
returndata=[]
ticker=['BTC-USD','ETH-USD','USDT-USD','ADA-USD','XRP-USD','SOL-USD']
for crypto in ticker:
    df=web.Datareader(crypto,'yahoo',date(2020,1,1),date(2020,12,31))['Adj Close']
    cryptoreturn=np.log(df/df.shift(1))
    returndata=np.concat(cryptoreturn,axis=1)
returndata.to_csv('ceshi.csv',encoding='tuf-8')
print(returndata)
#如果不行,则用append()函数尝试：

ticker=['BTC-USD','ETH-USD','USDT-USD','ADA-USD','XRP-USD','SOL-USD']
for crypto in ticker:
    df=web.DataReader(crypto,'yahoo',date(2020,1,1),date(2020,12,31))['Adj Close'])
    cryptoreturn=np.log(df/df.shift(1))
    
'''



data = pd.read_csv('gushi.csv')
returns_annual = data.mean(numeric_only=True)*252 #计算年化收益率
cov_annual = data.cov()*252 #计算协方差矩阵
#print(cov_annual)
c=np.sum(data[""])
print()

#print(returns_annual,'\n',cov_annual)

number_of_assets = 6
portfolio_returns =[]
portfolio_volatility=[]
sharpe_ratio=[]
for stock in range(100000):
    weight=np.random.random(number_of_assets)
    weight=weight/(np.sum(weight))
    #print('weight最初：',weight)
    returns=np.dot(weight,returns_annual)
    volatility = np.sqrt(np.dot(weight.T, np.dot(cov_annual, weight)))  # 投资组合波动率
    portfolio_returns.append(returns) #append()用于在后面加个什么东西
    portfolio_volatility.append(volatility)#我就理解为直接把volatility的数据给portfolio volatility
    sharpe=(returns-0.03)/volatility
    sharpe_ratio.append(sharpe)
portfolio_returns=np.array(portfolio_returns)
portfolio_volatility=np.array(portfolio_volatility)


plt.style.use('seaborn-dark')
plt.figure(figsize=(10, 4))
plt.scatter(portfolio_volatility, portfolio_returns, c=sharpe_ratio,cmap='RdYlGn', edgecolors='black',marker='.')
plt.grid(True)
plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier of Portfolios')
#plt.savefig('/Users/harper/Desktop/2.png',dpi=500,bbox_inches = 'tight')


def statistics(weights):
    weights = np.array(weights)
    #print("weight测试：",weights)
    #此处我有疑虑，因为我算的是ln的收入，真实收入应该为Preturn x e^Preturn,有机会调整一下
    Preturn = np.sum(data.mean(numeric_only=True) * weights) * 252
    Pvolatility = np.sqrt(np.dot(weights.T, np.dot(data.cov() * 252, weights)))
    #print("Portfolio return: %Preturn"%Preturn,"Portfolio volatility: %Pvolatility"%Pvolatility) 这里我不明白为什么会导致后面代码错误，只是尝试print信息，奇怪
    return np.array([Preturn, Pvolatility, (Preturn-0.03)/ Pvolatility])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

bnds = tuple((0, 1) for x in range(number_of_assets))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = sco.minimize(min_func_sharpe, number_of_assets * [1. / number_of_assets,], method='SLSQP',  bounds=bnds, constraints=cons)
print(opts['x'].round(3)) #得到各股票权重
print(statistics(opts['x']).round(3)) #得到投资组合预期收益率、预期波动率以及夏普比率
plt.show()
