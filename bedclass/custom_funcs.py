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
    df = df.sort_values(by=['PSX'], ascending=True).reset_index()
    
    return df
