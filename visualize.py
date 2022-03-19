import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import ShuffleSplit
import sklearn.model_selection as curves


def values_occurence(df, column, max_columns=20, pie=False):
    plt.figure(figsize=(15, 5))
    plt.title('Most popular values in column: ' + column)
    if pie:
        plt.pie(df[column].value_counts()[:max_columns], labels=df[column].value_counts().keys()[
            :max_columns])
    else:
        plt.bar(df[column].value_counts().keys()[
            :max_columns], df[column].value_counts()[:max_columns])
        plt.xticks(rotation='vertical')
    plt.show()


def distribution(df, column, bins=50, log=False):
    plt.figure(figsize=(15, 10))
    plt.subplot(211)
    if log:
        plt.title('Column: ' + column + ' logarithmic distribution')
    else:
        plt.title('Column: ' + column + ' distribution')
    plt.hist(df[column], bins=bins, log=log)
    plt.xlabel(column)
    plt.subplot(212)
    plt.title("Column: " + column + ' distribution')
    sns.boxplot(x=df[column])
    plt.show()


def boxplot_price(df, column, result_column='Log_Cena', figsize=(15, 5), orient='h'):
    plt.figure(figsize=figsize)
    sns.boxplot(x=result_column, y=column, data=df, orient=orient,
                order=df[column].value_counts().keys())
    plt.show()


def modelLearning(model, X, y, n_splits=5, scoring='rmse'):
    # Create cross-validtion sets for training and testing
    cv = ShuffleSplit(test_size=0.2,
                      random_state=0, n_splits=n_splits)

    # Generate the training set sizes
    train_sizes = np.rint(np.linspace(
        X.shape[0] * 0.2, X.shape[0] * 0.8 - 1, 9)).astype(int)

    # Create the figure window
    plt.figure(figsize=(5, 3))

    if scoring == 'rmse':
        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(model, X, y,
                                                                 cv=cv, train_sizes=train_sizes, scoring='neg_mean_squared_error', n_jobs=3)
        train_scores = np.sqrt(-train_scores)
        test_scores = np.sqrt(-test_scores)
    else:
        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(model, X, y,
                                                                 cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=3)

    # Find the mean and standard deviation for smoothing
    train_std = np.std(train_scores, axis=1)
    train_mean = np.mean(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot the learning curve
    plt.plot(sizes, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(sizes, test_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.15, color='g')

    # Labels
    plt.xlabel('Number of Training Points')
    plt.ylabel('RMSE')
    plt.xlim([X.shape[0]*0.2, X.shape[0]*0.8])
    plt.legend()
    plt.show()

    # Summary
    print('Model: ', model)
    print('RMSE(train_set): ', train_mean[-1])
    print('RMSE(val_set): ', test_mean[-1])


def modelParameter(model, param_name, param_range, X, y, n_splits=5, log=False, test_size=0.2):

    # Create cross-validation sets  for training and testing
    cv = ShuffleSplit(test_size=test_size,
                      random_state=0, n_splits=n_splits)

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(model, X, y,
                                                        param_name=param_name, param_range=param_range, cv=cv, scoring='neg_mean_squared_error', n_jobs=3)

    train_scores = np.sqrt(-train_scores)
    test_scores = np.sqrt(-test_scores)

    # Find the mean and standard deviation for smoothing
    train_std = np.std(train_scores, axis=1)
    train_mean = np.mean(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(5, 3))
    plt.title(str(model) + ' ' + param_name + ' Performance')
    plt.plot(param_range, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(param_range, test_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(param_range, test_mean - test_std,
                     test_mean + test_std, alpha=0.15, color='g')

    if log == True:
        plt.xscale('log')

    # Visual aesthetics
    plt.legend()
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.show()

    return train_mean, test_mean
