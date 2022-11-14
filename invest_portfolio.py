import asyncio
import matplotlib.pyplot as plt

from invest_portfolio_models import MarkovModel, SharpModel, StatModel
from PortfolioFilters import MakeBetaPositivePortfolio, IncomeTickerFilter
from pandas import DataFrame
from threading import Thread, ThreadError
from os import mkdir, chdir, path
from os.path import exists
from shutil import rmtree
from sys import exit
from warnings import filterwarnings
filterwarnings('ignore')
global DF_CLOSE_TICKERS
DIR_NAME = 'Graph_tickers'

if exists(DIR_NAME):
    rmtree(path.join(path.abspath(path.dirname(__file__)), DIR_NAME))
    mkdir(DIR_NAME)
    chdir(DIR_NAME)
else:
    mkdir(DIR_NAME)
    chdir(DIR_NAME)

try:
    import pandas_datareader as pdr
except ImportError as error:
    print(f"{error} at pandas_datareader")
    exit(1)


async def plot_df(df_close, tickers):
    fig = plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-whitegrid')
    for ticker in tickers:
        plt.plot(df_close[ticker], label=f'{ticker} цена закрытия')
    plt.ylabel('RUB')
    plt.xlabel('Дата')
    plt.grid(True)
    plt.legend()
    fig.savefig('All_tickers.png')


async def plot_daily_income(df_close, tickers):
    df_daily_return = df_close.copy()
    for i in df_close.columns[:]:
        for j in range(1, len(df_close)):
            df_daily_return[i][j] = ((df_close[i][j] - df_close[i][j - 1]) / df_close[i][j - 1]) * 100
        df_daily_return[i][0] = 0
    fig = plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-whitegrid')
    for ticker in tickers:
        plt.plot(df_daily_return[ticker], label=f'{ticker} дневная доходность')
    plt.title('Доходность акций')
    plt.ylabel('RUB')
    plt.xlabel('Индекс дня')
    plt.grid(True)
    plt.legend()
    fig.savefig('Daily_income_all_tickers.png')


async def parse_df_ticker(ticker, date_start, date_end):
    close_price = []
    df = pdr.data.DataReader(ticker, 'moex', date_start, date_end)['CLOSE']
    for price in df:
        close_price.append(price)
    return close_price


async def get_df_close_tickers(tickets, date_start, date_end):
    global DF_CLOSE_TICKERS
    result = DataFrame(columns=tickets)
    for ticker in tickets:
        result[ticker] = await parse_df_ticker(ticker, date_start, date_end)
    await plot_df(result, tickets)
    await plot_daily_income(result, tickets)
    DF_CLOSE_TICKERS = result


def thread_realise(func, *args):
    try:
        Thread(target=func, args=(*args,)).run()
    except ThreadError as Error:
        print(f"{Error} in {func.__name__}")
        exit(1)


def main():
    tickers = ["SBER", "GAZP", "YNDX", "ROSN", "VTBR", "POLY", "GMKN", "SNGS", "NVTK",
               "MGNT", "AFKS", "PLZL", "VKCO", "CHMF", "RSTI"]
    date_start, date_end = "2022-05-01", "2022-10-01"
    print("=================================================================================================")
    print("if you want to use BetaPositivePortfolio enter --- 1, if you want to use IncomeTickerFilter enter --- 2")
    filter_method = int(input("{+} Enter type filter = "))
    if filter_method == 1:
        new_tickers = MakeBetaPositivePortfolio(tickers, date_start, date_end).BetaPositivePortfolio(tickers,
                                                                                                     date_start,
                                                                                                     date_end)
        asyncio.run(get_df_close_tickers(new_tickers, date_start, date_end))
        print("\nПоследние значения цен по тикерам.")
        print(DF_CLOSE_TICKERS.tail(), '\n')
        thread_realise(MarkovModel(DF_CLOSE_TICKERS).__call__())
        print('\n')
        thread_realise(SharpModel(DF_CLOSE_TICKERS).__call__())
        thread_realise(StatModel(DF_CLOSE_TICKERS, tickers).__call__())
        print('\n\n')
        exit(0)
    if filter_method == 2:
        tickers = IncomeTickerFilter(tickers, date_start, date_end).income_filter()
        asyncio.run(get_df_close_tickers(tickers, date_start, date_end))
        print("\nПоследние значения цен по тикерам.")
        print(DF_CLOSE_TICKERS.tail(), '\n')
        thread_realise(MarkovModel(DF_CLOSE_TICKERS).__call__())
        print('\n')
        thread_realise(SharpModel(DF_CLOSE_TICKERS).__call__())
        thread_realise(StatModel(DF_CLOSE_TICKERS, tickers).__call__())
        print('\n\n')
        exit(0)
    else:
        print("{-} Type filter not found")
        exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt.")
        exit(1)
