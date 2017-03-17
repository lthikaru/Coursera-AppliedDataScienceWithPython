import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


os.chdir(os.path.join(os.getcwd(), ('coursera\\applied_data_science_with_python\\' +
                                    'applied_plotting_charting_and_data_representation_in_python')))


# read data, remove 29th of february and pivot the dataframe
df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv',
                 parse_dates=['Date']).set_index('Date')
df = df[~((df.index.day == 29) & (df.index.month == 2))].reset_index()
df['Data_Value'] *= 0.1
df['day_of_year'] = pd.to_datetime(df['Date'].map(lambda el: '1900-' + el.strftime('%m-%d')))
df_pivot = pd.pivot_table(df, values='Data_Value', index=['ID', 'Date', 'day_of_year'], columns='Element').reset_index()

# group by date
before_2015 = (df_pivot[df_pivot['Date'] < pd.to_datetime('2015-01-01')].groupby('day_of_year').
               agg({'TMAX': max, 'TMIN': min}).reset_index().sort_values('day_of_year'))

# plot
plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
xaxis = before_2015['day_of_year'].as_matrix()
plt.plot(xaxis, before_2015['TMIN'].as_matrix(), '-', xaxis, before_2015['TMAX'].as_matrix(), '-')
plt.gca().fill_between(xaxis,
                       before_2015['TMIN'].as_matrix(), before_2015['TMAX'].as_matrix(),
                       facecolor='blue',
                       alpha=0.25)

# add 2015 outliers
year_2015 = pd.merge(df_pivot[df_pivot['Date'] >= pd.to_datetime('2015-01-01')], before_2015, 'left', on='day_of_year')
plt.plot(year_2015.loc[(year_2015['TMIN_x'] < year_2015['TMIN_y']), 'day_of_year'].as_matrix(),
         year_2015.loc[(year_2015['TMIN_x'] < year_2015['TMIN_y']), 'TMIN_x'].as_matrix(), 'o', color='b')
plt.plot(year_2015.loc[(year_2015['TMAX_x'] > year_2015['TMAX_y']), 'day_of_year'].as_matrix(),
         year_2015.loc[(year_2015['TMAX_x'] > year_2015['TMAX_y']), 'TMAX_x'].as_matrix(), 'o', color='b')

# add legend
plt.legend(['Min $^o$C (2004-2014)', 'Max $^o$C (2004-2014)', '2015 temperatures'], frameon=False)

# format x axis ticks dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))
for item in plt.gca().xaxis.get_ticklabels():
    item.set_ha('left')
    item.set_rotation(-20)

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

# add title
plt.title('Temperatures ($^oC$) around Ann Arbor, Michigan, United States.\nClimate band is measured between 2005 and'
          ' 2014 while the dots are 2015 observations.')

# show plot
plt.show()