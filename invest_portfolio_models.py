import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.metrics import r2_score
from asyncio import run


class MarkovModel:
    def __init__(self, df_close):
        self.df_close = df_close
        self.df_close_data = df_close.pct_change()
        self.df_close_mean = self.df_close_data.mean()
        self.cov_matrix = self.df_close_data.cov()
        self.tickers_amount = len(self.df_close.columns)

    def random_portfolio(self):
        result = np.exp(np.random.randn(self.tickers_amount))
        result = result / result.sum()
        return result

    def profitability_of_portfolio(self, random_port):
        return np.matmul(self.df_close_mean.values, random_port)

    def risk_of_portfolio(self, random_port):
        return np.sqrt(np.matmul(np.matmul(random_port, self.cov_matrix.values), random_port))

    async def result(self):
        iterations = 10000
        risk = np.zeros(iterations)
        doh = np.zeros(iterations)
        portf = np.zeros((iterations, self.tickers_amount))

        for it in range(iterations):
            r = self.random_portfolio()
            portf[it, :] = r
            risk[it] = self.risk_of_portfolio(r)
            doh[it] = self.profitability_of_portfolio(r)

        fig = plt.figure(figsize=(10, 8))
        plt.style.use('seaborn-whitegrid')

        plt.scatter(risk * 100, doh * 100, c='y', marker='.')
        plt.xlabel('риск, %')
        plt.ylabel('доходность, %')
        plt.title("Облако портфелей")

        min_risk = np.argmin(risk)
        plt.scatter([(risk[min_risk]) * 100], [(doh[min_risk]) * 100], c='r', marker='*', label='минимальный риск')

        max_sharp_koef = np.argmax(doh / risk)
        plt.scatter([risk[max_sharp_koef] * 100], [doh[max_sharp_koef] * 100], c='g', marker='o',
                    label='максимальный коэффициент Шарпа')

        r_mean = np.ones(self.tickers_amount) / self.tickers_amount
        risk_mean = self.risk_of_portfolio(r_mean)
        doh_mean = self.profitability_of_portfolio(r_mean)
        plt.scatter([risk_mean * 100], [doh_mean * 100], c='b', marker='x', label='усредненный портфель')

        plt.legend()
        fig.savefig('Облако_портфелей.png')
        print('============= Портфель по Маркову =============')
        print('============= Минимальный риск =============', "\n")
        print("риск = %1.2f%%" % (float(risk[min_risk]) * 100.))
        print("доходность = %1.2f%%" % (float(doh[min_risk]) * 100.), "\n")
        print(DataFrame([portf[min_risk] * 100], columns=self.df_close.columns, index=['доли, %']).T, "\n")
        print('============= Максимальный коэффициент Шарпа =============', "\n")
        print("риск = %1.2f%%" % (float(risk[max_sharp_koef]) * 100.))
        print("доходность = %1.2f%%" % (float(doh[max_sharp_koef]) * 100.), "\n")
        print(DataFrame([portf[max_sharp_koef] * 100], columns=self.df_close.columns, index=['доли, %']).T, "\n")
        print('============= Средний портфель =============', "\n")
        print("риск = %1.2f%%" % (float(risk_mean) * 100.))
        print("доходность = %1.2f%%" % (float(doh_mean) * 100.), "\n")
        print(DataFrame([r_mean * 100], columns=self.df_close.columns, index=['доли, %']).T, "\n")
        print('=======================================', '\n')

    def __call__(self, *args, **kwargs):
        run(self.result())


class SharpModel:
    def __init__(self, df_close):
        self.df_close = df_close
        self.beta = {}
        self.alpha = {}
        self.ri = {}

    def daily_income(self, df_close):
        df_daily_return = df_close.copy()
        for i in df_close.columns[1:]:
            for j in range(1, len(df_close)):
                df_daily_return[i][j] = ((df_close[i][j] - df_close[i][j - 1]) / df_close[i][j - 1]) * 100
            df_daily_return[i][0] = 0
        return df_daily_return

    async def result(self):
        df_daily_return = self.daily_income(self.df_close)
        y = list(range(0, len(self.df_close)))
        res = dict()
        keys = self.df_close.columns
        print('============= Портфель по Шарпу =============\n')
        for key in keys:
            self.beta[key] = np.cov(self.df_close[key], y)[0][1] / np.var(self.df_close[key])
            self.alpha[key] = (sum(y) / len(y)) - self.beta[key] * (sum(self.df_close[key]) / len(self.df_close[key]))
            self.ri[key] = self.alpha[key] + (self.beta[key] * sum(df_daily_return[key]))
            if self.ri[key] > 0:
                res[key] = self.ri[key]
            # print('=================================================\n')
            # print('ticker({}) --- alpha({}), beta({})'.format(key, round(self.alpha[key], 2), round(self.beta[key]), 2))
            # print('r_i({}) --- {}'.format(key, round(self.ri[key]), 2))
            # print('\n=================================================')
        print('============= Пропорции по портфелю =============')
        for key in res.keys():
            print('{} --- {}%'.format(key, round(res[key] / sum(res.values()), 3) * 100))

    def __call__(self, *args, **kwargs):
        run(self.result())


class StatModel:
    def __init__(self, df_close, tickers):
        self.df_close = df_close
        self.tickers = tickers
        self.keys = df_close.columns
        self.mean_tickers = []
        self.std_tickers = []

    def get_mean_std_arr(self, df_close):
        for key in self.keys:
            self.mean_tickers.append(sum(df_close[key]) / len(df_close[key]))
            self.std_tickers.append(np.std(df_close[key]))

    async def result(self):
        self.get_mean_std_arr(self.df_close)
        res = DataFrame({"mean": self.mean_tickers, "std": self.std_tickers, "tickers": self.tickers})
        fig = plt.figure(figsize=(10, 8))
        plt.style.use('seaborn-whitegrid')
        plt.scatter(res['mean'], res['std'], marker='o')
        z = np.polyfit(res['mean'], res['std'], 1)
        y_hat = np.poly1d(z)(res['mean'])

        plt.plot(res['mean'], y_hat, "r--", lw=1, label='line of trend')
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(res['std'], y_hat):0.3f}$"
        plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')
        plt.xlabel('средняя цена за акцию')
        plt.ylabel('среднеквадратичное отклонение')
        plt.grid(True)
        ax = plt.gca()
        res.apply(lambda x: ax.annotate(x['tickers'], (x['mean'] + 0.2, x['std'])), axis=1)
        fig.savefig('Stat.png')

    def __call__(self, *args, **kwargs):
        run(self.result())
