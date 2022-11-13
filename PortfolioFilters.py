# Choosing a Beta Positive Portfolio
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from pandas import DataFrame
from sys import exit
from asyncio import run
from warnings import filterwarnings
filterwarnings('ignore')


class MakeBetaPositivePortfolio:
    def __init__(self, tickers_list, date_start, date_end):
        self.tickers_list = tickers_list
        self.date_start = date_start
        self.date_end = date_end
        self.beta = lambda df: round(float(df.cov()['SMA(5)'][1]) / float(df.std()['SMA(5)']), 3)
        self.tickers_dict_price = dict()
        self.tickers_list_result = list()

    async def plotting_ticker(self, df, ticker):
        fig = plt.figure(figsize=(10, 8))
        plt.style.use('seaborn-whitegrid')
        plt.subplot(2, 1, 1)
        plt.plot(df['CLOSE'], label='цена закрытия')
        plt.plot(df['SMA(5)'], label='SMA(5) цена закрытия')
        plt.title(f"{ticker}")
        plt.xlabel('Дата')
        plt.ylabel('Прогнозируемое значение')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(df['STD'], label='STD цена закрытия')
        plt.plot(df['DIFF'], label='Цена закрытия - Цена открытия')
        plt.xlabel('Дата')
        plt.grid(True)
        plt.legend()
        fig.savefig(f"{ticker}.png")

    async def parsing_ticker(self, ticker, date_start, date_end):
        try:
            new_df = DataFrame(columns=['SMA(5)', 'STD', 'DIFF'])
            df_ticker = pdr.data.DataReader(ticker, 'moex', date_start, date_end)
            df_ticker['SMA(5)'] = df_ticker['CLOSE'].rolling(5).mean()
            df_ticker['STD'] = df_ticker['CLOSE'].rolling(5).std()
            df_ticker['DIFF'] = df_ticker['CLOSE'] - df_ticker['OPEN']
            await self.plotting_ticker(df_ticker, ticker)
            new_df['SMA(5)'] = df_ticker['SMA(5)']
            new_df['STD'] = df_ticker['STD']
            new_df['DIFF'] = df_ticker['DIFF']
            return new_df
        except ImportError as Error:
            print(Error)
            exit(1)

    async def parsing_tickers_list_price(self, tickers_list, date_start, date_end):
        for ticker in tickers_list:
            self.tickers_dict_price[ticker] = await self.parsing_ticker(ticker, date_start, date_end)

    def BetaPositivePortfolio(self, tickers_list, date_start, date_end):
        run(self.parsing_tickers_list_price(tickers_list, date_start, date_end))
        for key in self.tickers_dict_price:
            print(f'beta = {self.beta(self.tickers_dict_price[key])} --- {key}')
            if self.beta(self.tickers_dict_price[key]) > 0:
                self.tickers_list_result.append(key)
        return self.tickers_list_result


class IncomeTickerFilter(MakeBetaPositivePortfolio):
    def __init__(self, tickers_list, date_start, date_end):
        super().__init__(tickers_list, date_start, date_end)

    def income_filter(self):
        filter_result, pos = [], 0
        run(self.parsing_tickers_list_price(self.tickers_list, self.date_start, self.date_end))
        values = [sum(self.tickers_dict_price[key]['DIFF']) for key in self.tickers_dict_price.keys()]
        mean = sum(values) / len(values)
        for key in self.tickers_dict_price.keys():
            print(f'income_filter value = {values[pos]}, income_filter mean = {mean}, key = {key}')
            if values[pos] > mean:
                filter_result.append(key)
            pos += 1
        return filter_result

