import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import six

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(33500,150000,3650),
                   np.random.normal(41000,90000,3650),
                   np.random.normal(41000,120000,3650),
                   np.random.normal(48000,55000,3650)],
                  index=[1992,1993,1994,1995])


def get_colormap(color_start, color_mid, color_end):
    color_start_rgb = mcolors.hex2color(color_start)
    color_end_rgb = mcolors.hex2color(color_end)
    midpoint_rgb = mcolors.hex2color(color_mid)

    color_dict = {'red': ((0, color_start_rgb[0], color_start_rgb[0]), (0.5, midpoint_rgb[0], midpoint_rgb[0]),
                          (1, color_end_rgb[0], color_end_rgb[0])),
                  'green': ((0, color_start_rgb[1], color_start_rgb[1]), (0.5, midpoint_rgb[1], midpoint_rgb[1]),
                            (1, color_end_rgb[1], color_end_rgb[1])),
                  'blue': ((0, color_start_rgb[2], color_start_rgb[2]), (0.5, midpoint_rgb[2], midpoint_rgb[2]),
                           (1, color_end_rgb[2], color_end_rgb[2]))}

    return mcolors.LinearSegmentedColormap('colors', color_dict, COL_CUTS)


def confidence_mean(x_normal):
    mean = x_normal.mean()
    distance = 1.96 * x_normal.std() / x_normal.size ** 0.5
    return mean - distance, mean, mean + distance


def normalize(x, lower, upper):
    return min((max(lower, x) - lower)/((upper - lower) * 1.0), 1)


def plot(y_value):

    # transform data
    df_plot = pd.DataFrame([val for val in zip(*df.apply(confidence_mean, axis=1))],
                           index=['5%', 'mean', '95%'], columns=df.index).transpose()

    # make a figure ready
    color_bar = plt.contourf([[0, 0], [0, 0]], [1.0*val / COL_CUTS for val in range(0, COL_CUTS + 1, 1)], cmap=COLORS)

    # plot
    plt.clf()
    plt.colorbar(color_bar, orientation='horizontal')
    bars = plt.bar(df_plot.index, df_plot['mean'],
                   width=0.8,
                   tick_label=df.index,
                   align='center',
                   alpha=1)

    # remove ticks
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

    # remove box edges
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # set background color
    plt.gca().set_axis_bgcolor('lightgrey')

    # change layout to look better
    plt.tight_layout()

    # add horizontal line
    plt.plot([df_plot.index.min() - 0.5, df_plot.index.max() + 0.5], [y_value] * 2, '-',
             color='black', linewidth=2, alpha=0.8)
    plt.xlim([df_plot.index.min() - 0.5, df_plot.index.max() + 0.5])

    # add confidence intevals
    for year in df_plot.index:
        plt.plot([year]*2, df_plot.loc[year, ['5%', '95%']], '-', color='black', linewidth=8, alpha=0.8)

    # change color of bars
    for bar in bars:
        col_idx = COL_CUTS - int(normalize(y_value,
                                           *df_plot.loc[int(bar.get_x()) + 1, ['5%', '95%']].tolist()) * COL_CUTS)
        rgb = COLORS(col_idx)
        bar.set_color(rgb)
        bar.set_edgecolor('black')

    plt.show()


def onclick(event):
    plot(event.ydata)

# set up some definitions
COL_CUTS = 20
BLUE = '#000080'
WHITE = '#ffffff'
RED = '#8b0000'
COLORS = get_colormap(BLUE, WHITE, RED)

# plot
plt.figure()
plot(38000)

# add interactivity
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
