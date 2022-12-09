from bedclass.config import *

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


def make_cmap_greyscale():
    import numpy as np
    from matplotlib.colors import ListedColormap

    cmap = np.zeros([256, 4])
    cmap[:, 3] = np.linspace(0, 1, 256)
    cmap = ListedColormap(cmap)
    return cmap


def read_UTG_line(infile):
    import pandas as pd
    
    df = pd.read_csv(infile)
    
    cols = df.columns[df.dtypes.eq('object')]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce') 
   
    df = df.sort_values(by=['PSX'], ascending=True).reset_index()
    df = calc_dist_projected(df, 'PSX', 'PSY')

    df['classvalue'] = df['classvalue'].fillna(0)
    
    return df


def line_overplot_3(df, my_vars, var_labels, x_var, x_label, my_colors, my_title, my_file, use_markers=False, prediction=False):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=80)

    ax.plot(df[x_var]/1e3, df[my_vars[0]], 
            color=my_colors[0])
    ax.tick_params(axis='y', labelcolor=my_colors[0])
    ax.set_ylabel(var_labels[0], 
                  color=my_colors[0], fontsize=16)
    ax.set_xlabel(x_label, color='black', fontsize=16)
    # plt.axvline(x=27, color="goldenrod", linestyle="--")
    add_basin_locs(ax)

    ax2 = ax.twinx()
    ax2.grid(False)
    if use_markers == True:
        ax2.plot(df[x_var]/1e3, df[my_vars[1]], 
                 color=my_colors[1], ls='', marker='.')
    else:
        ax2.plot(df[x_var]/1e3, df[my_vars[1]], 
                 color=my_colors[1])
    ax2.tick_params(axis='y', labelcolor=my_colors[1])
    ax2.set_ylabel(var_labels[1], color=my_colors[1], fontsize=16)
    ax2.spines['right'].set_position(('axes', 1.2))

    ax3 = ax.twinx()
    ax3.grid(False)
    if use_markers == True:
        ax3.plot(df[x_var]/1e3, df[my_vars[2]], 
                 color=my_colors[2], ls='', marker='*')
    else:
        ax3.plot(df[x_var]/1e3, df[my_vars[2]], color=my_colors[2])
    ax3.tick_params(axis='y', labelcolor=my_colors[2])
    ax3.set_ylabel(var_labels[2], color=my_colors[2], fontsize=16)
    
    if prediction == True:
        ax4 = ax.twinx()
        ax4.grid(False)
        ax4.plot(df[x_var].loc[df['predicted_class'] == 1]/1e3, df['predicted_class'].loc[df['predicted_class'] == 1], 
                 color='orange', ls='', marker='v', label='predicted basin')
        ax4.axes.get_yaxis().set_visible(False)

    plt.suptitle(my_title)
    plt.tight_layout()
    plt.savefig(my_file)
    plt.show()
    

def create_model_inputs(df, features, target, scale=True):
    from sklearn.model_selection import train_test_split
    
    ##
    model_cols = features + target
    
    ##
#     df_train = df.copy()
#     df_train = df_train[model_cols]
#     df_train[target] = df_train[target].astype('int')
    df_train = df[model_cols]
    df_train[target] = df_train[target].astype('int')
    
    ##
    if scale == True:
        from sklearn.preprocessing import StandardScaler
        
        sc = StandardScaler()

#         df_scaled = df_train.copy()
#         df_scaled[features] = sc.fit_transform(df_train[features])
        df_train[features] = sc.fit_transform(df_train[features])
    
        # temp = sc.fit_transform(df_train)
        # df_scaled = pd.DataFrame(temp, index=df_train.index, columns=df_train.columns)

    return df_train

    
def plot_model_inputs_all(df, features, target):
    import matplotlib.pyplot as plt
    from yellowbrick.features import Rank2D
    from yellowbrick.features import radviz
    from yellowbrick.features import parallel_coordinates
    from yellowbrick.features import pca_decomposition
    from yellowbrick.features import JointPlotVisualizer

    
    fig, ax = plt.subplots(dpi=80)  # figsize=(8, 8), 
    ax.set_aspect('equal')
    
    ## 1
    visualizer = Rank2D(algorithm="pearson")
    visualizer.fit_transform(df[features])  
    visualizer.show()
    
    ## 3
    visualizer = JointPlotVisualizer(columns=['boug', 'mag'])
    visualizer.fit_transform(df, df[target].values)
    visualizer.show()

    ## 2
    visualizer = parallel_coordinates(df[features], 
                                      df[target].values.squeeze(), normalize="standard")
    visualizer.show()    

    ## 4
    visualizer = radviz(df[features], 
                        df[target].values.squeeze(), colors=["maroon", "gold"])
    visualizer.show()
    
    ## 5
    visualizer = pca_decomposition(df[features].values,
                                   df[target].astype('int').values.squeeze())
    visualizer.show()

    
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    
    return diagonal_sum / sum_of_all_elements

    
def run_ML_model(df, features, target, model_type='MLP', model_score=True, model_plots=False):
    from sklearn.model_selection import train_test_split

    ## Splitting the dataset into  training and validation sets
    training_set, validation_set = train_test_split(df, test_size = 0.2, random_state = 21)
#     X_train = training_set[feature_cols].values
#     y_train = training_set[target_col].values.ravel()
#     X_test = validation_set[feature_cols].values
#     y_test = validation_set[target_col].values.ravel()

    ## MODEL
    if (model_type == 'MLP'):
        from sklearn.neural_network import MLPClassifier
        
        classifier = MLPClassifier(hidden_layer_sizes=(), alpha = 0,  # =(150,100,50) 
                                   max_iter=300, 
                                   activation = 'relu', 
                                   solver='lbfgs',  # 'lbfgs' 'adam'
                                   random_state=1)
        # classifier = GaussianNB()
        classifier.fit(training_set[features].values, 
                       training_set[target].values.ravel())
        y_hat = classifier.predict(validation_set[features].values)
        
    elif (model_type == 'Perceptron'):
        from sklearn.linear_model import Perceptron

        classifier = Perceptron(tol=1e-3, random_state=0)
        classifier.fit(training_set[features].values, 
                       training_set[target].values.ravel())
        coeffs = classifier.coef_
        y_hat = classifier.predict(validation_set[features].values)
        for cid, c in enumerate(df[features].columns):
            print(f"{c:12}: {coeffs[0, cid]:5.2f}") 
            
    elif (model_type == 'LogisticRegression'):    
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression()
        classifier.fit(training_set[features].values, 
               training_set[target].values.ravel())
        y_hat = classifier.predict(validation_set[features].values)
    else:
        print("Sorry we don't have that model at this time.")
        
    ## SCORES   
    if model_score == True:
        ##
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_hat, validation_set[target].values.ravel())
        print(f"Accuracy of MLPClassifier : {accuracy(cm)}")
        
        ##
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import ShuffleSplit
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

        scores = cross_val_score(classifier, 
                        training_set[features].values, 
                        training_set[target].values.ravel(), 
                        cv=cv,
                        scoring='f1_macro',
                        )
        print("\nShuffleSplit cross-validation:")
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    
    ## PLOTS
    if model_plots == True:
        import matplotlib.pyplot as plt
        from yellowbrick.classifier import ClassificationReport
        from yellowbrick.classifier import ROCAUC

        fig, ax = plt.subplots(dpi=80)  # figsize=(8, 8), 
        ax.set_aspect('equal')

        ## Instantiate the classification model and visualizer
        visualizer = ClassificationReport(classifier, 
                                          classes=["not basin", "basin"], 
                                          support=True)

        visualizer.fit(training_set[features].values, 
                   training_set[target].values.ravel())        # Fit the visualizer and the model
        visualizer.score(validation_set[features].values, 
                         validation_set[target].values.ravel())        # Evaluate the model on the test data
        visualizer.show()                       # Finalize and show the figure
        
        ##
        visualizer = ROCAUC(classifier, iterations=500, binary=True)
#         visualizer = ROCAUC(classifier)
        
        visualizer.fit(training_set[features].values, 
                   training_set[target].values.ravel())
        visualizer.score(validation_set[features].values, 
                         validation_set[target].values.ravel())
        visualizer.show()

    y_hats = classifier.predict(df[features].values)
#     df['y_pred'] = y_hats
    
    return y_hats, classifier, scores


def read_ASE_csv(infile, features):
    import pandas as pd
        
    df = pd.read_csv(infile)  
#     df = df_ASE.copy()
    df.rename({'bouguer': 'boug'}, axis=1, inplace=True)
#     df = df[features]
    
    return df
     

def read_shapefile(infile):
    import sys
    import geopandas as gpd
    import pyproj

    # fix PROJ path
    projpath = sys.prefix + '/share/proj'
    pyproj.datadir.set_data_dir(projpath)
    shapefile = gpd.read_file(infile)

    return shapefile
    
    
def plot_ML_map(df, var_plot, fileout):
    import os
    import matplotlib.pyplot as plt

    cmap = make_cmap_greyscale()
    shapefile = read_shapefile(os.path.join(get_project_root(), 'data/external/ASE_catchments+GL_3031.shp'))

    fig, ax = plt.subplots(dpi=80)  # figsize=(8, 8),
    ax.set_aspect('equal')
    ##
    shapefile.boundary.plot(ax=ax, edgecolor='black')
    ##
    plt.scatter(df.X, df.Y, c=df[var_plot], 
        marker=',',
#             vmin=-60, vmax=60,
        cmap="terrain", # "Spectral_r"
        )  
    plt.colorbar(label="BedMachine Bed [m.a.s.l.]")
    ##
    plt.scatter(df.X, df.Y, c=df['predicted_class'], 
                marker='.',
                cmap=cmap, edgecolors=None)
    
    ## don't have original line df...
#     colors= ['chocolate' if l == 0 else 'gold' for l in df['y_pred']]
#     plt.scatter(df.PSX, df.PSY, c=colors,
#                 marker='.', edgecolors=None)

    plt.xlim(-1.8e6, -1e6)
    plt.ylim(-1e6, -0e6)
    plt.savefig(fileout)
    plt.show()
    
    
def predict_csv(classifier, target_datafile, features, model_plots_maps=True):
    from sklearn.preprocessing import StandardScaler
    
    ## Read in features for ASE from csv
    df = read_ASE_csv(target_datafile, features)
    
    ## Scale
    df_scaled = df.copy()
    df_scaled[features] = StandardScaler().fit_transform(df[features])

    ## Fill NA
    # df_ASE_scaled['water'] = df_ASE_scaled['water'].fillna(0)
    df_scaled.fillna(value=0, inplace=True)
    
    ## Interpolat NaNs, then Predict
#     df['bedmachine'].interpolate(method='polynomial', order=2).plot()
#     df['boug'].interpolate(method='linear').plot()
    y_pred_ASE = classifier.predict(df_scaled[features].values)
    
    ## Save as Pandas DF
    # df_out = pd.merge(df,y_test[['preds']],how = 'left',left_index = True, right_index = True)
    df['predicted_class'] = y_pred_ASE

    ##
    if model_plots_maps:
        var_plot = 'bedmachine'
        file_plot_ML_map = f'figs/mapplot_ASE_ypred_{var_plot}_script.png'
        plot_ML_map(df, var_plot, file_plot_ML_map)

    return df

    


