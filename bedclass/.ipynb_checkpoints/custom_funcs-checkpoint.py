def add_basin_locs(ax):

    ax.axvspan(21.4, 33.3, alpha=0.5, color='goldenrod')
    ax.axvspan(72.15, 94.45, alpha=0.5, color='goldenrod')
    ax.axvspan(124.73, 140.82, alpha=0.5, color='goldenrod')
    ax.axvspan(153.70, 160.65, alpha=0.5, color='goldenrod')
    ax.axvspan(173.03, 183.53, alpha=0.5, color='goldenrod')

    
def define_basin_locs(df, sed_column='classvalue'):
    df[sed_column].iloc[0] = df[sed_column].iloc[1]
    # df['sed_basin'].iloc[0] = df['sed_basin'].iloc[1]
    
    x_list = (df['Dist'].iloc[df[df[sed_column].diff() != 0].index.tolist()]/1e3).values
    x_pair = list(zip(x_list, x_list[1:] + x_list[:1]))

    return x_pair[1:]
    
    
def calc_dist_projected(df, xvar='PSX', yvar='PSY'):
    import numpy as np
    
    df['Dist'] = 0
    for i in range(1, len(df)-1):
        #     dfmax.remove(columnsn='Dist')
        #     print('{}, {}'.format(dfmax['LAT'].iloc[i-1], dfmax['LON'].iloc[i-1]))
        #     print('{}, {}'.format(dfmax['LAT'].iloc[i], dfmax['LON'].iloc[i]))
        df['Dist'].iloc[i] = \
        df['Dist'].iloc[i-1] + np.sqrt((df[xvar].iloc[i-1] - df[xvar].iloc[i])**2 \
                  - (df[yvar].iloc[i-1] - df[yvar].iloc[i])**2)
    
        # print(df['PSX'].iloc[i-1] - df['PSX'].iloc[i])
        # print(df['PSY'].iloc[i-1] - df['PSY'].iloc[i])
        # print('{} Dist: {}'.format(i, df['Dist'].iloc[i]))

    # df['Dist'][df['Dist'] > 2*df['Dist'].mean()] = df['Dist'].mean()
    df['Dist'].iloc[0] = 0
    df['Dist'].iloc[-1] = df['Dist'].iloc[-2] + df['Dist'].iloc[:-2].diff().mean()
    
    return df


def read_UTG_line(infile):
    import pandas as pd
    
    df = pd.read_csv(infile)

    cols = df.columns[df.dtypes.eq('object')]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    df = df.sort_values(by=['PSX'], ascending=True).reset_index()
    df = calc_dist_projected(df, 'PSX', 'PSY')
    
    return df


def line_overplot_3(df, my_vars, var_labels, x_var, x_label, my_colors, my_file, use_markers=False):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=80)

    ax.plot(df[x_var]/1e3, df[my_vars[0]], color=my_colors[0])
    ax.tick_params(axis='y', labelcolor=my_colors[0])
    ax.set_ylabel(var_labels[0], color=my_colors[0], fontsize=16)
    ax.set_xlabel(x_label, color='black', fontsize=16)
    # plt.axvline(x=27, color="goldenrod", linestyle="--")
    add_basin_locs(ax)

    ax2 = ax.twinx()
    if use_markers == True:
        ax2.plot(df[x_var]/1e3, df[my_vars[1]], color=my_colors[1], ls='', marker='s')
    else:
        ax2.plot(df[x_var]/1e3, df[my_vars[1]], color=my_colors[1])
    ax2.tick_params(axis='y', labelcolor=my_colors[1])
    ax2.set_ylabel(var_labels[1], color=my_colors[1], fontsize=16)
    ax2.spines['right'].set_position(('axes', 1.2))

    ax3 = ax.twinx()
    if use_markers == True:
        ax3.plot(df[x_var]/1e3, df[my_vars[2]], color=my_colors[2], ls='', marker='*')
    else:
        ax3.plot(df[x_var]/1e3, df[my_vars[2]], color=my_colors[2])
    ax3.tick_params(axis='y', labelcolor=my_colors[2])
    ax3.set_ylabel(var_labels[2], color=my_colors[2], fontsize=16)

    plt.tight_layout()
    plt.savefig(my_file)
    plt.show()
