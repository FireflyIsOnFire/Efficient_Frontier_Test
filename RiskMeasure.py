import numpy as np
from scipy import stats
import math

class RiskMeasure():
    def __init__(self,daily_change, days):
        self.daily_change = daily_change
        self.days = days

    def VaR_95p_parameter(self):  # problem is: it's not log-return.
        # days: horizon of VaR
        mu = self.daily_change.mean()
        std = self.daily_change.std()
        VaR = mu * self.days - 1.645 * std * np.sqrt(self.days)
        # print(mu,'\n',std)
        n_test_result = stats.kstest(self.daily_change, 'norm', (mu, std))  # not normal distribution cuz p<0.5
        print(n_test_result, '\n', 'if p-value<0.05, this model isnt normal distribution! VaR-parameter unsuitable!')
        return VaR

    def VaR_MCS(self):
        # days: horizon of VaR
        mu = self.daily_change.mean()
        std = self.daily_change.std()
        price_changes = []
        n_test_result = stats.kstest(self.daily_change, 'norm', (mu, std))  # not normal distribution cuz p<0.5
        print(n_test_result, '\n',
              'if p-value<0.05, this model is not a normal distribution! VaR-parameter unsuitable!')
        for i in range(1000):
            dis = np.random.normal(loc=mu, scale=std, size=None)
            price_changes.append(dis)
        price_changes = np.array(price_changes)
        VaR = price_changes.mean() * self.days - 1.645 * price_changes.std() * np.sqrt(self.days)
        return VaR

    def Expected_Shortfall_95p(self):
        mu = self.daily_change.mean()
        std = self.daily_change.std()
        ES = -(mu * self.days + np.sqrt(self.days) * std * (
                    math.e ** (-(1.645 ** 2) / 2) / ((1 - 0.95) * np.sqrt(2 * 3.1415926))))
        return ES

    def Historical_Risk_95p(self):
        length = round(len(self.daily_change) * 0.05)
        daily_change = self.daily_change[np.argsort(self.daily_change)]
        # print(daily_change)
        summary = 0
        for i in range(length):
            summary = daily_change[i].astype(np.float) + summary
        summary = summary / length
        return summary
