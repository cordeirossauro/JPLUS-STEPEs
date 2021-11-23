'''
Assorted functions used in the machine learning codes during the Blue Stars project

Created by: Vinicius Cordeiro (viniciuscordeiro@on.br)
'''
#External packages and functions used
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from sklearn.metrics import (mean_absolute_error, r2_score, max_error, mean_squared_error)

from  itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import time
#Sets of filters and corrections used
Filters = {'JPLUS': ['uJAVA', 'J0378', 'J0395', 'J0410', 'J0430', 'gSDSS', 
                     'J0515', 'rSDSS', 'J0660', 'iSDSS', 'J0861', 'zSDSS'],
           'WISE': ['W1', 'W2', 'W3', 'W4'],
           'GALEX': ['NUVmag'],
           'GAIA': ['G', 'BP', 'RP']
          }

Corrections = {'JPLUS': [('Ax_' + Filter) for Filter in Filters['JPLUS']]}


def MagnitudeCorrection(df, FilterSet, CorrectionSet, NewDF):    
    '''
    Correct the magnitudes of a set of filters inside a dataframe
    
    Keyword Arguments:
    df - Dataframe with uncorrected magnitudes
    FilterSet - Set of filters to correct
    CorrectionSet - Set of corrections
    NewDF - If True, a new dataframe is returned with just the corrected values;
            If False, the function returns the complete original dataframe with the uncorrected values replaced by the corrected ones.
    '''
    
    if NewDF == True:
        TempDF = pd.DataFrame()
        for Index in range(0, len(FilterSet)):
            TempDF[FilterSet[Index]] = df[FilterSet[Index]] - df[CorrectionSet[Index]]
        
        return TempDF
    else:
        for Index in range(0, len(FilterSet)):
            df[FilterSet[Index]] = df[FilterSet[Index]] - df[CorrectionSet[Index]]
        
        return df


def CreateColors(df, FilterSet, NewDF):
    '''
    Create all the possible filter combinations (colors) for a set of filters inside a dataframe
    
    Keyword arguments:
    df - Dataframe with the magnitudes
    FilterSet - Set of filters to combine
    NewDF - If True, a new dataframe is returned with just the combined values;
            If False, the function returns the complete original dataframe with the combinations added.
    '''
    
    CombinationsList = list(combinations(FilterSet, 2))
    
    if NewDF == True:
        TempDF = pd.DataFrame()
        for Combination in CombinationsList:
            CombinationName = '(' + Combination[0] + ' - ' + Combination[1] + ')'
            TempDF[CombinationName] = (df[Combination[0]] - df[Combination[1]])
        
        return TempDF
    
    else:
        for Combination in CombinationsList:
            CombinationName = '(' + Combination[0] + ' - ' + Combination[1] + ')'
            df[CombinationName] = (df[Combination[0]] - df[Combination[1]])
        
        return df
        

def CreateCombinations(df, FilterSet, NewDF):
    '''
    Create all the possible color combinations for a set of filters inside a dataframe
    
    Keyword arguments:
    df - Dataframe with the magnitudes
    FilterSet - Set of filters to combine
    NewDF - If True, a new dataframe is returned with just the combined values;
            If False, the function returns the complete original dataframe with the combinations added.
    '''
    
    CombinationsList = list(combinations(FilterSet, 4))
    
    if NewDF == True:
        TempDF = pd.DataFrame()
        for Combination in CombinationsList:
            CombinationName = '(' + Combination[0] + ' - ' + Combination[1] + ') - (' + Combination[2] + ' - ' + Combination[3] + ')'
            TempDF[CombinationName] = (df[Combination[0]] - df[Combination[1]]) - (df[Combination[2]] - df[Combination[3]])
        
        return TempDF
    
    else:
        for Combination in CombinationsList:
            CombinationName = '(' + Combination[0] + ' - ' + Combination[1] + ') - (' + Combination[2] + ' - ' + Combination[3] + ')'
            df[CombinationName] = (df[Combination[0]] - df[Combination[1]]) - (df[Combination[2]] - df[Combination[3]])
        
        return df

def ReindexDF(df):
    '''
    Create a column by combining the TILE_ID and the NUMBER of each star and set it as the new index of the dataframe
    
    Keyword arguments:
    df - Dataframe to reindex
    '''
    
    #Function to create the new column
    def CreateID(df):
        return (str(df[0]) + ', ' + str(df[1]))
    
    #Create a new column with the TILE_ID and the NUMBER to identify each star individually
    df['ID NUMBER'] = df[['TILE_ID', 'NUMBER']].apply(CreateID, axis = 1)
    #Set the ID NUMBER column as the new index
    df = df.set_index('ID NUMBER', drop = False)
    #Drop any duplicates
    df = df.drop_duplicates(subset = ['ID NUMBER'])
    
    return df
    
def AssembleWorkingDF(df, addWISE, addGALEX, addGAIA, Colors, Combinations):
    '''
    Assemble a dataframe with JPLUS magnitudes and, when asked, colors and color combinations.
    
    Keyword arguments:
    df - Dataframe with the magnitudes
    addWISE - If True, WISE filters will also be used to make the colors and combinations
    Colors - If True, all the possible colors will be added to the returnde dataframe
    Combinations - If True all the possible color combinations will be added to the returned dataframe
    '''
    
    #Set the index of the dataframe as a combination of the TILE_ID and NUMBER of each star
    df = ReindexDF(df)
    magnitudes_df = df[Filters['JPLUS']]
    
    #If asked for, the WISE filters are added to the dataframe
    if addWISE is True:
        magnitudes_df = pd.concat([magnitudes_df, df[Filters['WISE']]], axis = 1)
    
    #If asked for, the GALEX filters are added to the dataframe
    if addGALEX is True:
        magnitudes_df = pd.concat([magnitudes_df, df[Filters['GALEX']]], axis = 1)
        
    if addGAIA is True:
        magnitudes_df = pd.concat([magnitudes_df, df[Filters['GAIA']]], axis = 1)
        
    WorkingDF = magnitudes_df
    
    #If asked for, create a dataframe with all the possible colors and add it to the working dataframe
    if Colors is True:
        ColorsDF = CreateColors(magnitudes_df, magnitudes_df.columns, NewDF = True)
        WorkingDF = pd.concat([WorkingDF, ColorsDF], axis = 1)
        
    #If asked for, create a dataframe with all the possible color combinations and add it to the working dataframe
    if Combinations is True:
        CombinationsDF = CreateCombinations(magnitudes_df, magnitudes_df.columns, NewDF = True)
        WorkingDF = pd.concat([WorkingDF, CombinationsDF], axis = 1)
    
    #Return the resulting dataframe
    return (df, WorkingDF)


def NeuralNetRegressor(InputSize, Layers, learning_rate, beta1, beta2, epsilon):
    '''
    Create, compile and return a regressor neural network with a specified input size and internal layer structure.
    
    Keyword arguments:
    InputSize - Number of features given to the network as input
    LayerStructure - A list where each item is the number of nodes in the respective internal layer (e.g., for a network with
                     two layers of 8 nodes each, the list passed would be LayerStructure = [8, 8])
    learning_rate - Learning rate to be used on the network optimizer
    '''
    # Initialize the neural Network object
    NeuralNet = Sequential()
    
    # Add the input and first internal layers
    NeuralNet.add(Dense(Layers[0], input_dim = InputSize, 
                        activation = tf.keras.layers.LeakyReLU()))
    
    # Start a loop to add all the other internal layers
    for LayerSize in Layers[1:]:
    # For each item in the LayerStructure list, add a new layer
        NeuralNet.add(Dense(LayerSize, activation = tf.keras.layers.LeakyReLU()))
    
    # Add the output layer
    NeuralNet.add(Dense(1, activation = 'linear'))
    
    # Compile the Neutal Network with the mean squared error as the loss metric and the adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = beta1, beta_2 = beta2, epsilon = epsilon)
    NeuralNet.compile(loss='mse', optimizer=optimizer)
    
    # Return the compiled Neural Network
    return NeuralNet


def rf_evaluator(HyperParams, X, y, n_splits, n_repeats, verbose = 0):
    '''
    Evaluate a STEPE-RF model (Feature Selector + Random Forest) initialized with a
    certain combination of hyperparameters by using a k-fold n-repeat cross validation 
    and return the average values and standard deviations of a set of 
    metrics (MAE, RMSE, MaxError, R2 Score and elapsed time).
    
    Keyword arguments:
    Hyperparams - A list containing the combination of hyperpameters to test, in the 
                format [n_features, n_trees, min_samples_leaf, max_features, criterion]

    X - Dataframe containing the input values of the development sample 
        (will be split into training and validation samples)

    y - Dataframe containing the target values of the development sample 
        (will be split into training and validation samples)

    n_splits - Number of samples that the data will be split into during the k-fold, 
               n-repeat cross-validation (corresponds to k)

    n_repeats - Number of times that the data will be shuffled and split again during 
                the k-fold, n-repeat cross-validation (corresponds to n)

    verbose - Indicates whether the function will print information on the screen or not
    '''

    # Get the hyperparameters from the list
    n_features, n_trees, min_samples_leaf, max_features, criterion = HyperParams
    
    # If asked for, print a message indicating the combination beign evaluated
    if verbose == 1:
        init_message = ('Starting Random Forest {}-Fold, {} repeat CV with: n_features = {}, ' + 
                        'n_trees = {}, min_samples_leaf = {} , max_features = {}, criterion = {}...')
        print(init_message.format(n_splits, n_repeats, n_features, n_trees, min_samples_leaf, max_features, criterion))
        
    # Initialize a list for each error metric and the time
    MAEs = []
    RMSEs = []
    MaxEs = []
    R2s = []
    Times = []
    
    # Initialize the cross-validation function
    KFSplitter = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats)
    
    # Initialize a variable to count the current iteration
    Index = 0
    
    # Loop through all the training/validation combinations given by the cross-validation function
    for train_index, validation_index in KFSplitter.split(X):
        Index = Index + 1
    
        # If asked for, print a message indication the current iteration
        if verbose == 1:
            print('Current Fold: {}/{}'.format(Index, n_splits * n_repeats))
        
        # Get the training and validation samples
        x_train, x_validation = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
        
        # Initialize the feature selector using the given hyperparameters
        FeatureSelector = RFE(estimator=DecisionTreeRegressor(), 
                              n_features_to_select = n_features, 
                              verbose = 0, step = 20)
    
        # Initialize the random forest using the given hyperparameters
        RF = RandomForestRegressor(n_estimators=n_trees, 
                                   min_samples_leaf = min_samples_leaf, 
                                   max_features=max_features, 
                                   criterion=criterion)
        
        # Create a pipeline with the feature selector and the random forest
        RFPipeline = Pipeline(steps = [('Feature Selector', FeatureSelector),('Model', RF)])
        
        # Start a timer to time the process
        StartTime = time.time()
        
        # Fit the pipeline to the training data
        RFPipeline = RFPipeline.fit(x_train, y_train.values.reshape(len(y_train)))
        
        # Predict the target values of the validation sample
        Predictions = RFPipeline.predict(x_validation)
        
        # Stop the timer
        EndTime = time.time() 
        
        # Calculate the error metrics and append them to their lists
        MAE = mean_absolute_error(y_validation, Predictions)
        RMSE = np.sqrt(mean_squared_error(y_validation, Predictions))
        MaxE = max_error(y_validation, Predictions)
        R2 = r2_score(y_validation, Predictions)
        
        MAEs.append(MAE)
        RMSEs.append(RMSE)
        MaxEs.append(MaxE)
        R2s.append(R2)
        
        # Append the value of the elapsed time to its list
        Times.append(EndTime - StartTime)
    
    # Calculate the mean and standard deviation for all the error metrics
    MAEs = np.array(MAEs)
    MeanMAE = MAEs.mean()
    StdMAE = MAEs.std()
    
    RMSEs = np.array(RMSEs)
    MeanRMSE = RMSEs.mean()
    StdRMSE = RMSEs.std()
    
    MaxEs = np.array(MaxEs)
    MeanMaxE = MaxEs.mean()
    StdMaxE = MaxEs.std()
    
    R2s = np.array(R2s)
    MeanR2 = R2s.mean()
    StdR2 = R2s.std()
    
    Times = np.array(Times)
    MeanTime = Times.mean()
    StdTime = Times.std()
    
    # If asked for, print the results to the screen
    if verbose == 1:
        print('CV Process Finished! Results:')
        print('Mean Absolute Error: {:.3f} ({:.3f})'.format(MeanMAE, StdMAE))
        print('Root Mean Squared Error: {:.3f}  ({:.3f})'.format(MeanRMSE, StdRMSE))
        print('Max Error: {:.3f} ({:.3f})'.format(MeanMaxE, StdMaxE))
        print('R2 Score: {:.3f} ({:.3f})'.format(MeanR2, StdR2))
        print('Time Elapsed: {:.3f} ({:.3f}) s'.format(MeanTime, StdTime))
        print('\n')

    # Return the results
    return MeanMAE, StdMAE, MeanRMSE, StdRMSE, MeanMaxE, StdMaxE, MeanR2, StdR2, MeanTime, StdTime


def nn_evaluator(HyperParams, X, y, n_splits, n_repeats, verbose = 0):
    '''
    Evaluate a STEPE-NN model (Scaler + Feature Selector + Neural Network) initialized 
    with a certain combination of hyperparameters by using a k-fold n-repeat 
    cross-validation and return the average values and standard deviations of a set of 
    metrics (MAE, RMSE, MaxError, R2 Score and elapsed time).
    
    Keyword arguments:
    Hyperparams - A list containing the combination of hyperpameters to test, in the 
                format [FeaturesSize, Epochs, BatchSize, learning_rate, 
                        beta1, beta2, epsilon, structure]

    X - Dataframe containing the input values of the development sample 
        (will be split into training and validation samples)

    y - Dataframe containing the target values of the development sample 
        (will be split into training and validation samples)

    n_splits - Number of samples that the data will be split into during the k-fold, 
               n-repeat cross-validation (corresponds to k)

    n_repeats - Number of times that the data will be shuffled and split again during 
                the k-fold, n-repeat cross-validation (corresponds to n)

    verbose - Indicates whether the function will print information on the screen or not
    '''

    # Get the hyperparameters from the list
    FeaturesSize, Epochs, BatchSize, learning_rate, beta1, beta2, epsilon, structure  = HyperParams

    # If asked for, print a message indicating the combination beign evaluated
    if verbose == 1:
        init_message = ('Starting Neural Network {}-Fold, {} repeat CV process with: FeaturesSize = {}, ' +
                        'Epochs = {}, BatchSize = {}, LearningRate = {}, Beta1 = {}, ' +
                        'Beta2 = {}, epsilon = {}, structure = {}')
                
        print(init_message.format(n_splits, n_repeats, FeaturesSize, Epochs, BatchSize, learning_rate, beta1, beta2, epsilon, structure))

    # Initialize a list for each error metric and the time
    MAEs = []
    RMSEs = []
    MaxEs = []
    R2s = []
    Times = []

    # Initialize the cross-validation function
    KFSplitter = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats)

    # Initialize a variable to count the current iteration
    Index = 0

    # Loop through all the training/validation combinations given by the cross-validation function
    for train_index, validation_index in KFSplitter.split(X):
        Index = Index + 1

        # If asked for, print a message indication the current iteration
        if verbose == 1:
            print('Current Fold: {}/{}'.format(Index, int(n_splits * n_repeats)))

        # Get the training and validation samples
        x_train, x_validation = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]

        # Start a timer to time the process
        StartTime = time.time()

        # Initialize the standard scaler
        Scaler = StandardScaler()

        # Fit the scaler to the training data
        Scaler.fit(x_train)

        # Scale the training and validation data
        x_train = Scaler.transform(x_train)
        x_validation = Scaler.transform(x_validation)

        # Initialize the feature selector using the given hyperparameters
        FeatureSelector = RFE(estimator=DecisionTreeRegressor(),
                              n_features_to_select = FeaturesSize,
                              verbose = 0, step = 200)

        # Fit the feature selector to the training data
        FeatureSelector.fit(x_train, y_train)

        # Transform the training data according with the fitted feature selector
        x_train = FeatureSelector.transform(x_train)
        x_validation = FeatureSelector.transform(x_validation)

        # Initialize the neural network with the given hyperparameters
        NeuralNet = NeuralNetRegressor(FeaturesSize, structure, learning_rate, beta1, beta2, epsilon)

        # Fit the neural network to the training data using the given hyperparameters
        NeuralNet.fit(x=x_train, y=y_train.values,
                      validation_data=(x_validation, y_validation.values),
                      batch_size=BatchSize, epochs=Epochs, verbose=0)

        # Predict the target values of the validation sample and reshape the result
        Predictions = NeuralNet.predict(x_validation)
        Predictions = Predictions.reshape(len(Predictions))

        # Stop the timer
        EndTime = time.time()

        # Calculate the error metrics and append them to their lists
        MAE = mean_absolute_error(y_validation, Predictions)
        RMSE = np.sqrt(mean_squared_error(y_validation, Predictions))
        MaxE = max_error(y_validation, Predictions)
        R2 = r2_score(y_validation, Predictions)
        
        MAEs.append(MAE)
        RMSEs.append(RMSE)
        MaxEs.append(MaxE)
        R2s.append(R2)

        # Append the value of the elapsed time to its list
        Times.append(EndTime - StartTime)

    # Calculate the mean and standard deviation for all the error metrics
    MAEs = np.array(MAEs)
    MeanMAE = MAEs.mean()
    StdMAE = MAEs.std()
    
    RMSEs = np.array(RMSEs)
    MeanRMSE = RMSEs.mean()
    StdRMSE = RMSEs.std()
    
    MaxEs = np.array(MaxEs)
    MeanMaxE = MaxEs.mean()
    StdMaxE = MaxEs.std()
    
    R2s = np.array(R2s)
    MeanR2 = R2s.mean()
    StdR2 = R2s.std()
    
    Times = np.array(Times)
    MeanTime = Times.mean()
    StdTime = Times.std()

    # If asked for, print the results to the screen
    if verbose == 1:
        print('CV Process Finished! Results:')
        print('Mean Absolute Error: {:.3f} ({:.3f})'.format(MeanMAE, StdMAE))
        print('Root Mean Squared Error: {:.3f}  ({:.3f})'.format(MeanRMSE, StdRMSE))
        print('Max Error: {:.3f} ({:.3f})'.format(MeanMaxE, StdMaxE))
        print('R2 Score: {:.3f} ({:.3f})'.format(MeanR2, StdR2))
        print('Time Elapsed: {:.3f} ({:.3f}) s'.format(MeanTime, StdTime))
        print('\n')
    
    # Return the results
    return MeanMAE, StdMAE, MeanRMSE, StdRMSE, MeanMaxE, StdMaxE, MeanR2, StdR2, MeanTime, StdTime


def create_heatmap_matrix(data, conditions, x_axis, y_axis, value_label):
    '''
    Create a dataframe containing a certain type of value and with the given x and y axes 
    by applying a set of conditions to another dataframe
    
    Keyword arguments:
    data - Dataframe from which the final dataframe will be created

    conditions - Dictionary containing a set of conditions that, when applied to the
                 dataframe, results in a len(x_axis) x len(y_axis) dataframe. The format
                 for this dict is {"variable_1": value_1, "variable_2": value_2}

    x_axis - Dictionary containing the name of the variable and a list of values to use
             as the x-axis. The format for this dict is {"variable_x": [x_1, x_2, ...]}

    y_axis - Dictionary containing the name of the variable and a list of values to use
             as the y-axis. The format for this dict is {"variable_y": [y_1, y_2, ...]}

    value_label - The label of the variable that will be inside the final dataframe
    '''

    # Get the label and the values of each axis
    x_label, x_values = list(x_axis.items())[0]
    y_label, y_values = list(y_axis.items())[0]

    # Initialize an empty dataframe that will receive the values
    heatmap = pd.DataFrame(index = y_values,
                           columns = x_values,
                           dtype = float)

    # Loop through all the given conditions and apply them to the initial dataframe
    for condition_label in conditions:
        condition_value = conditions[condition_label]
        data = data[data[condition_label] == condition_value]

    # Loop through all the positions inside the final dataframe
    for x_value in x_values:
        for y_value in y_values:

            # For each position, get the corresponsing value inside the initial dataframe
            matrix_value = data[(data[x_label] == x_value) &
                                (data[y_label] == y_value)][value_label]

            # If there is a value inside the initial dataframe, put it inside the final
            # one, and if there isn't, put 0.
            try:
                heatmap.at[y_value, x_value] = matrix_value.values[0]
            except:
                heatmap.at[y_value, x_value] = 0

    # Return the final dataframe
    return heatmap


def plot_heatmaps(left_top_matrix, left_top_label,
                  left_bottom_matrix, left_bottom_label,
                  right_top_matrix, right_top_label,
                  right_bottom_matrix, right_bottom_label,
                  x_label, y_label,
                  value_format, cmap, v_min, v_max, 
                  colorbar_label, colorbar_ticks):

    hp_space_heatmap = plt.figure(figsize=(10.0, 10.0))
    hp_space_heatmap.patch.set_facecolor('white')
    
    left_top_ax = hp_space_heatmap.add_axes([0.075, 0.575, 0.38, 0.375])
    left_bottom_ax = hp_space_heatmap.add_axes([0.075, 0.075, 0.38, 0.375])
    right_top_ax = hp_space_heatmap.add_axes([0.475, 0.575, 0.38, 0.375])
    right_bottom_ax = hp_space_heatmap.add_axes([0.475, 0.075, 0.38, 0.375])
    
    cbar_ax = hp_space_heatmap.add_axes([0.9, 0.075, 0.025, 0.875])

    left_top_heatmap = sns.heatmap(data = left_top_matrix, vmin = v_min, vmax = v_max, ax = left_top_ax, cbar = 1, cbar_ax = cbar_ax, 
                                   cbar_kws = {'ticks':np.linspace(v_min, v_max, colorbar_ticks)}, linewidths=1.0,
                                   annot = True, fmt = value_format, annot_kws = {'size': 14}, cmap = cmap)
    left_top_ax.set_xticklabels(left_top_matrix.columns, size = 14)
    left_top_ax.set_yticklabels(left_top_matrix.index, rotation = 0, size = 14)
    left_top_ax.set_title(left_top_label, fontsize = 20, pad = 5)
    left_top_ax.set_xlabel(x_label, fontsize = 18)
    left_top_ax.set_ylabel(y_label, fontsize = 18)

    left_bottom_heatmap = sns.heatmap(data = left_bottom_matrix, vmin = v_min, vmax = v_max, ax = left_bottom_ax, cbar = 0, 
                                      linewidths=1.0, annot = True, fmt = value_format, annot_kws = {'size': 14}, cmap = cmap)
    left_bottom_ax.set_xticklabels(left_bottom_matrix.columns, size = 14)
    left_bottom_ax.set_yticklabels(left_bottom_matrix.index, rotation = 0, size = 14)
    left_bottom_ax.set_title(left_bottom_label, fontsize = 20, pad = 5)
    left_bottom_ax.set_xlabel(x_label, fontsize = 18)
    left_bottom_ax.set_ylabel(y_label, fontsize = 18)
    
    right_top_heatmap = sns.heatmap(data = right_top_matrix, vmin = v_min, vmax = v_max, ax = right_top_ax, cbar = 0,
                                    annot = True, yticklabels = False, fmt = value_format, linewidths=1.0,
                                    annot_kws = {'size': 14}, cmap = cmap)
    right_top_ax.set_xticklabels(right_top_matrix.columns, size = 14)
    right_top_ax.set_title(right_top_label, fontsize = 20, pad = 5)
    right_top_ax.set_xlabel(x_label, fontsize = 18)
    right_top_ax.set_ylabel('', fontsize = 0)
    
    right_bottom_heatmap = sns.heatmap(data = right_bottom_matrix, vmin = v_min, vmax = v_max, ax = right_bottom_ax, cbar = 0,
                                       annot = True, yticklabels = False, fmt = value_format, linewidths=1.0,
                                       annot_kws = {'size': 14}, cmap = cmap)
    right_bottom_ax.set_xticklabels(right_bottom_matrix.columns, size = 14)
    right_bottom_ax.set_title(right_bottom_label, fontsize = 20, pad = 5)
    right_bottom_ax.set_xlabel(x_label, fontsize = 18)
    right_bottom_ax.set_ylabel('', fontsize = 0)
    
    cbar = left_top_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar_ax.set_title(colorbar_label, fontsize = 16, pad = 15)
    
    return hp_space_heatmap


def plot_heatmaps_single_line(left_matrix, left_label, right_matrix, right_label, x_label, y_label,
                              value_format, cmap, v_min, v_max, colorbar_label, colorbar_ticks):

    hp_space_heatmap = plt.figure(figsize=(10.0, 5.0))
    hp_space_heatmap.patch.set_facecolor('white')
    
    left_ax = hp_space_heatmap.add_axes([0.075, 0.125, 0.38, 0.77])
    right_ax = hp_space_heatmap.add_axes([0.475, 0.125, 0.38, 0.77])
    cbar_ax = hp_space_heatmap.add_axes([0.875, 0.125, 0.025, 0.77])

    left_heatmap = sns.heatmap(data = left_matrix, vmin = v_min, vmax = v_max, ax = left_ax, cbar = 1, cbar_ax = cbar_ax, 
                               cbar_kws = {'ticks':np.linspace(v_min, v_max, colorbar_ticks)}, linewidths=1.0,
                               annot = True, fmt = value_format, annot_kws = {'size': 14}, cmap = cmap)
    left_ax.set_xticklabels(left_matrix.columns, size = 14)
    left_ax.set_yticklabels(left_matrix.index, rotation = 0, size = 14)
    left_ax.set_title(left_label, fontsize = 20, pad = 15)
    left_ax.set_xlabel(x_label, fontsize = 18)
    left_ax.set_ylabel(y_label, fontsize = 18)

    
    right_heatmap = sns.heatmap(data = right_matrix, vmin = v_min, vmax = v_max, ax = right_ax, cbar = 0,
                                annot = True, yticklabels = False, fmt = value_format,  linewidths=1.0,
                                annot_kws = {'size': 14}, cmap = cmap)
    right_ax.set_xticklabels(left_matrix.columns, size = 14)
    right_ax.set_title(right_label, fontsize = 20, pad = 15)
    right_ax.set_xlabel(x_label, fontsize = 18)
    right_ax.set_ylabel('', fontsize = 0)
    
    
    cbar = left_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar_ax.set_title(colorbar_label, fontsize = 16, pad = 15)
    
    return hp_space_heatmap


def plot_test_graphs(y_test, predictions, parameter_label, parameter_range, error_range, color):

    test_graphs = plt.figure(1, figsize = [12.8, 6.4])
    test_graphs.patch.set_facecolor('white')

    predictions_plot = test_graphs.add_axes([0.1, 0.325, 0.45, 0.625])
    errors_plot = test_graphs.add_axes([0.1, 0.1, 0.45, 0.2])
    errors_distribution = test_graphs.add_axes([0.64, 0.1, 0.35, 0.85])


    predictions_plot.scatter(x = y_test, y = predictions, color = color, s = 2.5, alpha = 0.25)
    predictions_plot.plot([parameter_range[0], parameter_range[1]], [parameter_range[0], parameter_range[1]], color = 'k', linewidth = 1.25)
    predictions_plot.grid()

    predictions_plot.set_xlim(parameter_range[0], parameter_range[1])
    predictions_plot.tick_params(axis = 'x', labelsize = 0)

    predictions_plot.set_ylim(parameter_range[0], parameter_range[1])
    predictions_plot.tick_params(axis = 'y', labelsize = 16)
    predictions_plot.set_ylabel(r'Predicted ' + parameter_label, fontsize = 20, labelpad = 15)


    errors_plot.scatter(x = y_test, y = y_test - predictions, color = color, s = 2.5, alpha = 0.25)
    errors_plot.plot([parameter_range[0], parameter_range[1]], [0, 0], color = 'k', linewidth = 1.25)
    errors_plot.grid()

    errors_plot.set_xlim(parameter_range[0], parameter_range[1])
    errors_plot.tick_params(axis = 'x', labelsize = 16)
    errors_plot.set_xlabel(r'LAMOST ' + parameter_label, fontsize = 20)
    
    errors_plot.set_ylim(error_range[0], error_range[1])
    errors_plot.tick_params(axis = 'y', labelsize = 16)
    errors_plot.set_ylabel(r'Error', fontsize = 20)


    errors_distribution.hist(y_test - predictions, bins = 50, histtype = 'bar', ec = 'black', color = color)

    errors_distribution.set_xlim(error_range[0], error_range[1])
    errors_distribution.tick_params(axis = 'x', labelsize = 16)
    errors_distribution.set_xlabel(r'Error', fontsize = 20)

    errors_distribution.tick_params(axis = 'y', labelsize = 16)
    errors_distribution.set_ylabel(r'Count', fontsize = 20)
    
    return test_graphs
