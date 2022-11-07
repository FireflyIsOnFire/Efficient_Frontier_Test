import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import date
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import stats
import seaborn as sns
import math
from RiskMeasure import *

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus']=False
pd.set_option('display.max_columns',None)

begin = date(2020,1,1)
end = date.today()
interval = (end-begin).days
risk_free_rate = 0.04/365*interval

tickers = ['BTC-USD','ETH-USD','USDT-USD','ADA-USD','XRP-USD','SOL-USD','DOGE-USD','DOT-USD','DAI-USD','MATIC-USD']
df = pd.DataFrame()


for i in tickers:
    df[i] = web.DataReader(i,'yahoo',start = '2020-1-1',end = date.today())['Adj Close']
data = np.log(df/df.shift(1))
df_dc = df/df.shift(1)-1
data.to_csv('Database.csv')
data_corr = data.corr()


#main code

data = pd.read_csv('Database.csv')
returns_annual = data.mean(numeric_only=True)*interval #calculate mean return
cov_annual = data.cov()*interval #covariance matrix
number_of_assets = len(data.columns)-1


weight = []
optimal_search = []
here_to_find_ratio = []
for stock in range(1000000): #MCS
    next_i = False
    while True:
        weight = np.random.random(number_of_assets)
        weight = weight / (np.sum(weight))
        returns = np.dot(weight, returns_annual)
        volatility = np.sqrt(np.dot(weight.T, np.dot(cov_annual, weight)))  # portfolio volatility in interval
        sharpe = (returns - risk_free_rate) / volatility

        for re,vo in optimal_search:
            if (re > returns) & (vo < volatility):
                next_i = True
                break
        if next_i:
            break
        here_to_find_ratio.append([returns, volatility, sharpe, weight])
        optimal_search.append([returns,volatility])
        #weight.append(weight)

#print(optimal_search)
optimal_search = pd.DataFrame(optimal_search)
#print(optimal_search)
optimal_search.columns = ['re','vola']


#print(total_data[:,1])  会报错 tuple,, must be integers or slices
total_data = pd.DataFrame(here_to_find_ratio)
total_data.columns = ['returns','volatilities','SPI','weights']
total_data.to_csv('here_to_find_ratio')

#print(total_data)
#total_data = np.array(total_data)

optimal_search = optimal_search.sort_values(by = ['vola'])
#print(optimal_search)



# If short selling is available:

def record(weights):
    weights = np.array(weights)
    P_return = np.sum(data.mean(numeric_only=True) * weights) * interval
    P_volatility = np.sqrt(np.dot(weights.T, np.dot(data.cov() * interval, weights)))
    SPI = ((P_return - risk_free_rate) / P_volatility)
    #print('return: ',P_return, 'volatility: ', P_volatility, 'SPI: ',SPI,'\n')
    return np.array([P_return,P_volatility,SPI])

def func(weights):
    return -record(weights)[2]

#here shortselling is allowed, therefore the assigned weight could be negative
bnds = tuple((-1, 1) for x in range(number_of_assets))
x0 = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = opt.minimize(func, x0, method='SLSQP', bounds=bnds, constraints=cons)
print(opts,'\n')
print('Assets: ',tickers,'\n','Weights: ',(opts.x).round(3),'\n','returns:  Volatilities:  SPI: ','\n', record(opts['x']),'\n'*2)

#global optimizition: this function didn't need constraints and bounds.
opts_g = opt.basinhopping(func, x0, niter = 100)
#print('*'*50,'\n',opts_g)




# Rsik modeling
opt_weight = opts['x'].round(3)
port_daily_change = opt_weight*df_dc

port_daily_change.insert(len(port_daily_change.columns),'Portfolio',0)
port_daily_change['Portfolio'] = port_daily_change.apply(lambda x:x.sum(),axis=1)
port_daily_change.to_csv('Portfolio')


#Bootstrapping database
bootstrapping_data = port_daily_change['Portfolio']
sorted_data = np.array(bootstrapping_data)


risks = RiskMeasure(sorted_data,1)

# exam risk factors
a = risks.VaR_MCS()
b = risks.VaR_95p_parameter()
c = risks.Expected_Shortfall_95p()
d = risks.Historical_Risk_95p()
print('VaR with Monte-Carlo-Simulation: ',a,'\n','VaR with parameter: ',b,'\n','Expected Shortfall: ',c,'\n','Historical maximal loss: ',d)


#print(plt.style.available)
plt.figure(figsize=(8, 4))
plt.style.use('bmh')
plt.figure(1)
plt.grid(True)
heatmap = plt.subplot(221)
heatmap = sns.heatmap(data_corr, annot=True ,cmap='RdBu_r', vmin=-1, vmax=1)
plt2 = plt.subplot(212)
plt2 = plt.scatter(total_data['volatilities'], total_data['returns'], c=total_data['SPI'],cmap='RdYlGn', edgecolors='black',marker='.')
#plt.plot(optimal_search['vola'],optimal_search['re'])
plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.colorbar(label='Sharpe Ratio')
#plt.title('Efficient Frontier of Portfolios')

#plot dist
plt3 = plt.subplot(222)
plt3 = plt.hist(bootstrapping_data,bins=50, alpha=0.6, color='steelblue')
plt.ylabel("frequence", fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
#plt.title("Changes distribution & VaR/ES", fontsize=16)
plt.axvline(a,color='r',label='VaR with MCS')
plt.axvline(b,color='b',label='VaR with parameter')
plt.axvline(c,color='y',label='Expected Shortfall')
plt.axvline(d,color='g',label='Historical max. loss')
plt.legend()
plt.show()


