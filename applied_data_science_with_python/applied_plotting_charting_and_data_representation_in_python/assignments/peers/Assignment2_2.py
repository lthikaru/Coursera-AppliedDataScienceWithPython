import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime as dt
from textwrap import wrap

#Read in the data from the .csv file converting the 'Date' column into datetime format
df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv',
                parse_dates=['Date'])

#Take the 'Date' and 'Date_Value' columns and group by 'Date'. Aggregate using the max and min, which will generate the
#max and min daily values. Take the data only for the years 2005-2014.
df_max = df[['Date','Data_Value']].groupby('Date').apply(np.max).loc['2005-01-01':'2014-12-31',:]
df_min = df[['Date','Data_Value']].groupby('Date').apply(np.min).loc['2005-01-01':'2014-12-31',:]

#Add a 'Month' and 'Day' columns so that the data can be grouped in that way to obtain the daily max/min over the
#10-year period
df_max['Month'] = df_max['Date'].map(lambda x: x.month)
df_min['Month'] = df_max['Date'].map(lambda x: x.month)
df_max['Day'] = df_max['Date'].map(lambda x: x.day)
df_min['Day'] = df_min['Date'].map(lambda x: x.day)

#Exclude Feb. 29 from the data
df_max = df_max[~((df_max['Month'] == 2) & (df_max['Day'] == 29))]
df_min = df_min[~((df_min['Month'] == 2) & (df_min['Day'] == 29))]

#Group by 'Month','Day' and aggregate by max/min to generate the max/min over all the years (2005-2014)
df_max_final = df_max.groupby(['Month','Day']).apply(np.max)
df_min_final = df_min.groupby(['Month','Day']).apply(np.min)
df_min_final.drop('Date',1)
df_min_final['Date'] = df_max_final['Date']

#Get the start/end date that will be used to constrain the x-axis
dmin = df_max_final['Date'].values[0]
dmax = df_max_final['Date'].values[-1]

#Create a new dataframe that contains only the data for the last year (2015) and add 'Mnoth' and 'Day' columns
df_2015 = df[df['Date'] > pd.to_datetime('2014-12-31')]
df_2015['Month'] = df_2015['Date'].map(lambda x: x.month)
df_2015['Day'] = df_2015['Date'].map(lambda x: x.day)

#Create a groupby object that will contain the dataframes for each individual station (grouped on 'ID')
g = df_2015.sort(columns='Date').groupby('ID').groups

#Go through all of the station data and take the data that meets the condition of fallin outside the max/min
dfmax_list = []
dfmin_list = []
for station in g.keys():
    #Get the max/min values for the station
    df_max_station = df_2015.loc[g[station],:].groupby(['Month','Day']).max().drop(['Element','Date'],1)
    df_min_station = df_2015.loc[g[station],:].groupby(['Month','Day']).min().drop(['Element','Date'],1)
    dfmax = df_max_final.merge(df_max_station, left_index=True, right_index=True)
    dfmin = df_min_final.merge(df_min_station, left_index=True, right_index=True)
    dfmax_list.append(dfmax[dfmax['Data_Value_y'] > dfmax['Data_Value_x']])
    dfmin_list.append(dfmin[dfmin['Data_Value_y'] < dfmin['Data_Value_x']])

#Plotting the data
months = mdates.MonthLocator()
monthsFrmt = mdates.DateFormatter('%b')

fig = plt.figure()
#Plot the max daily values over the years 2005-2014
plt.plot(df_max_final['Date'].values,df_max_final['Data_Value'].values/10,c='dimgray',linewidth = 1, alpha=0.9)
#Plot the min daily values over the years 2005-2014
plt.plot(df_max_final['Date'].values,df_min_final['Data_Value'].values/10,c='dimgray',linewidth = 1, alpha = 0.9)
#Color the space between the first two plots
plt.gca().fill_between(df_max_final['Date'].values,df_min_final['Data_Value'].values/10,df_max_final['Data_Value'].values/10,
                      facecolor='gray',alpha=0.2)

plt.title("\n".join(wrap('$Max/Min$ daily temperatures between 2005-2014 and scatter plot of days in 2015 where they were exceeded. (Ann Arbor, Michigan, United States)',
                         60)))
plt.ylabel('Temperature ($\circ$C)',color='dimgray')
plt.gca().title.set_color('dimgray')
plt.gca().tick_params(axis='both',colors='dimgray')
plt.gca().spines['bottom'].set_color('dimgray')
plt.gca().spines['top'].set_color('dimgray')
plt.gca().spines['left'].set_color('dimgray')
plt.gca().spines['right'].set_color('dimgray')
plt.gca().xaxis.set_major_locator(months)
plt.gca().xaxis.set_major_formatter(monthsFrmt)
plt.gca().set_xlim(dmin,dmax)
plt.xticks(rotation=45)
for df_st in dfmax_list:
    plt.scatter(df_st['Date'].values,df_st['Data_Value_y'].values/10,s=3,c='indianred',alpha=0.9)
for df_st in dfmin_list:
    plt.scatter(df_st['Date'].values,df_st['Data_Value_y'].values/10,s=3,c='royalblue',alpha=0.5)
plt.show()
fig.tight_layout()
fig.savefig('W2.pdf', format='pdf', dpi = 330)


