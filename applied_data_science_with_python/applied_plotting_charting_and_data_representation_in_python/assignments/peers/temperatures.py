import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

non_leaping_year = 2017


def is_not_leap(series):
    ret = []
    for i in series:
        ret.append(not (i.month == 2 and i.day == 29))
    return ret

def plot(binsize, hash):
    df = pd.read_csv('data/C2A2_data/BinnedCsvs_d{}/{}.csv'.format(binsize, hash))
    df['Date'] = pd.to_datetime(df['Date'])
    df['Data_Value'] = df['Data_Value'].map(lambda x : x / 10.0)


    df['Month'] = df['Date'].map(lambda x : x.month)
    df['Day'] = df['Date'].map(lambda x : x.day)
    df['Year'] = df['Date'].map(lambda x : x.year)

    #filter leap days
    mask = is_not_leap(df['Date'])
    df = df[mask]

    #filter for years 2005 - 2014
    baseline = df[(df['Year'] >= 2005) & (df['Year'] <= 2014)]

    max_values = baseline.groupby(['Month', 'Day'])["Data_Value"].max().reset_index()
    max_values["id"] = pd.to_datetime(non_leaping_year * 10000 + max_values.Month * 100 + max_values.Day, format='%Y%m%d')
    max_values.set_index(["id"], inplace = True)
    max_values.drop(['Month', 'Day'], axis=1, inplace = True)
    max_values.rename(columns={'Data_Value': 'max_value'}, inplace = True)

    min_values = baseline.groupby(['Month', 'Day'])["Data_Value"].min().reset_index()
    min_values["id"] = pd.to_datetime(non_leaping_year * 10000 + min_values.Month * 100 + min_values.Day, format='%Y%m%d')
    min_values.set_index(["id"], inplace = True)
    min_values.drop(['Month', 'Day'], axis=1, inplace = True)
    min_values.rename(columns={'Data_Value': 'min_value'}, inplace = True)

    year2015 = df[(df['Year'] == 2015)]
    year2015max = year2015.groupby(['Month', 'Day'])["Data_Value"].max().reset_index()
    year2015max["id"] = pd.to_datetime(non_leaping_year * 10000 + year2015max.Month * 100 + year2015max.Day, format='%Y%m%d')
    year2015max.set_index(["id"], inplace = True)
    year2015max.drop(['Month', 'Day'], axis=1, inplace = True)
    year2015max.rename(columns={'Data_Value': '2015_max_value'}, inplace = True)

    year2015min = year2015.groupby(['Month', 'Day'])["Data_Value"].min().reset_index()
    year2015min["id"] = pd.to_datetime(non_leaping_year * 10000 + year2015min.Month * 100 + year2015min.Day, format='%Y%m%d')
    year2015min.set_index(["id"], inplace = True)
    year2015min.drop(['Month', 'Day'], axis=1, inplace = True)
    year2015min.rename(columns={'Data_Value': '2015_min_value'}, inplace = True)


    compare = pd.concat([max_values, min_values, year2015max, year2015min], axis = 1)
    print(compare.head())

    months = mdates.MonthLocator()  # every month
    monthsFmt = mdates.DateFormatter("%b")
    #monthsFmt = mdates.DateFormatter('%m')
    days = mdates.DayLocator()  # every day

    fig, ax = plt.subplots()

    baseline_min, = ax.plot(compare.index, compare["min_value"], color="#7570b3", alpha=0.5, label="min temperature 2005-2014")
    baseline_max, = ax.plot(compare.index, compare["max_value"], color="#d95f02", alpha = 0.5, label="max temperature 2005-2014")

    #year2015 higher outliers
    higher_outliers = compare[(compare['max_value'] < compare['2015_max_value'])]
    higher_outliers.drop(['max_value', 'min_value', '2015_min_value'], axis = 1, inplace = True)
    higher_outliers.rename(columns={'2015_max_value': 'value'}, inplace = True)

    lower_outliers = compare[(compare['min_value'] > compare['2015_min_value'])]
    lower_outliers.drop(['max_value', 'min_value', '2015_max_value'], axis = 1, inplace = True)
    lower_outliers.rename(columns={'2015_min_value': 'value'}, inplace = True)

    outliers = pd.concat([higher_outliers, lower_outliers])

    outliers_scatter, = ax.plot(higher_outliers.index, higher_outliers["value"], "o", color="#1b9e77", label="outliers")

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)

    ax.set_title('Temperature outliers in 2015', alpha=0.8)
    ax.set_xlabel('')
    ax.set_ylabel('Temperature in °C', alpha=0.8)

    ax.fill_between(min_values.index, compare["min_value"], compare["max_value"], color="lightgrey")
    #fig.autofmt_xdate()

    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handles=[baseline_min, baseline_max, outliers_scatter])

    #remove box top, right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #less emphasis on axis
    ax.spines['bottom'].set_alpha(0.8)
    ax.spines['left'].set_alpha(0.8)

    plt.show()



plot(400,'67cd5b7b956aa6979116b645539df874f6f79d18cdce8315e08117f8')
