import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from datetime import date
import numpy.random as npr
import matplotlib.pyplot as plt
from pylab import mpl
import scipy.optimize as sco
from scipy import stats
import seaborn as sns
import math

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus']=False
pd.set_option('display.max_columns',None)

begin = date(2020,1,1)
end = date.today()
interval = (end-begin).days
#print(interval)

tickers = ['BTC-USD','ETH-USD','USDT-USD','ADA-USD','XRP-USD','SOL-USD','DOGE-USD','DOT-USD','DAI-USD','MATIC-USD']
df = pd.DataFrame()


for i in tickers:
    df[i] = web.DataReader(i,'yahoo',start = '2020-1-1',end = date.today())['Adj Close']
data = np.log(df/df.shift(1))
df_dc = df/df.shift(1)-1
data.to_csv('Database.csv')




#Weird bug, I cant set end data except today(), have to collect all data then get the slice
'''
for i in tickers:
    df1[i] = web.DataReader(i,'yahoo',stress_begin, stress_end)['Adj Close']
stressed_data = df/df.shift(1)-1
stressed_data.to_csv('StressedData')

print(stress_interval)
print(stressed_data.head(stress_interval))

'''
#print(df_dc)
#print(df_r,'\n',data)



data_corr = data.corr()
#print(data_corr)

# correlation heatmap
sns.heatmap(data_corr, annot=True ,cmap='RdBu_r', vmin=-1, vmax=1)
#plt.show()




risk_free_rate = 0.04/365*interval


#main code

data = pd.read_csv('Database.csv')
returns_annual = data.mean(numeric_only=True)*interval #calculate mean return
cov_annual = data.cov()*interval #covariance matrix
#print(cov_annual)

#print(returns_annual,'\n',cov_annual)

number_of_assets = len(data.columns)-1
#print(number_of_assets)


portfolio_returns = []
portfolio_volatility = []
sharpe_ratio = []
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


#print(plt.style.available)
plt.style.use('bmh')
plt.figure(figsize=(8, 4))
plt.scatter(total_data['volatilities'], total_data['returns'], c=total_data['SPI'],cmap='RdYlGn', edgecolors='black',marker='.')
#plt.plot(optimal_search['vola'],optimal_search['re'])
plt.grid(True)
plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier of Portfolios')


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

bnds = tuple((-1, 1) for x in range(number_of_assets))
x0 = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = sco.minimize(func, x0, method='SLSQP', bounds=bnds, constraints=cons)
print('Assets: ',tickers,'\n','Weights: ',(opts.x).round(3),'\n','returns:  Volatilities:  SPI: ','\n', record(opts['x']),'\n'*2)






# Rsik modeling
opt_weight = opts['x'].round(3)
port_daily_change = opt_weight*df_dc
#print(port_daily_change)
#print(len(port_daily_change.columns))



port_daily_change.insert(len(port_daily_change.columns),'Portfolio',0)
#print(port_daily_change)

port_daily_change['Portfolio'] = port_daily_change.apply(lambda x:x.sum(),axis=1)
port_daily_change.to_csv('Portfolio')
#print(port_daily_change)

#Bootstrapping database
bootstrapping_data = port_daily_change['Portfolio']
#print(bootstrapping_data)



sorted_data = np.array(bootstrapping_data)


def VaR_95p_parameter(daily_change,days): #problem is: it's not log-return.
    # days: horizon of VaR
    mu = daily_change.mean()
    std = daily_change.std()
    VaR = mu*days-1.645*std*np.sqrt(days)
    #print(mu,'\n',std)
    n_test_result = stats.kstest(sorted_data, 'norm', (mu, std))  # not normal distribution cuz p<0.5
    print(n_test_result,'\n','if p-value<0.05, this model isnt normal distribution! VaR-parameter unsuitable!')
    return VaR


def VaR_MCS(daily_change,days):
    # days: horizon of VaR
    mu = daily_change.mean()
    std = daily_change.std()
    price_changes=[]
    n_test_result = stats.kstest(sorted_data, 'norm', (mu, std))  # not normal distribution cuz p<0.5
    print(n_test_result,'\n','if p-value<0.05, this model is not a normal distribution! VaR-parameter unsuitable!')
    for i in range(1000):
        dis = np.random.normal(loc=mu,scale=std,size=None)
        price_changes.append(dis)
    price_changes=np.array(price_changes)
    VaR = price_changes.mean()*days - 1.645*price_changes.std()*np.sqrt(days)
    return VaR

def Expected_Shortfall_95p(daily_change,days):
    mu = daily_change.mean()
    std = daily_change.std()
    ES = -(mu*days + np.sqrt(days)*std*(math.e**(-(1.645**2)/2)/((1-0.95)*np.sqrt(2*3.1415926))))
    return ES

def Historical_Risk_95p(daily_change):
    length = round(len(daily_change)*0.05)
    daily_change=daily_change[np.argsort(daily_change)]
    #print(daily_change)
    summary = 0
    for i in range(length):
        summary = daily_change[i].astype(np.float) + summary
    summary=summary/length
    return summary







# exam risk factors
a = VaR_MCS(sorted_data,1)
b = VaR_95p_parameter(sorted_data,1)
c = Expected_Shortfall_95p(sorted_data,1)
d = Historical_Risk_95p(sorted_data)
print('VaR with Monte-Carlo-Simulation: ',a,'\n','VaR with parameter: ',b,'\n','Expected Shortfall: ',c,'\n','Historical maximal loss: ',d)

#plot dist
plt.rcParams['figure.figsize']=(9,7)
f = plt.figure()
plt.hist(bootstrapping_data,bins=50, alpha=0.6, color='steelblue')
plt.ylabel("frequence", fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title("Changes distribution & VaR/ES", fontsize=16)
plt.axvline(a,color='r',label='VaR with MCS')
plt.axvline(b,color='b',label='VaR with parameter')
plt.axvline(c,color='y',label='Expected Shortfall')
plt.axvline(d,color='g',label='Historical max. loss')
plt.legend()
plt.show()


