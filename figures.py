import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def performance_attribution(model_aucs, feature_attributes, substrings, substring_names, feature_indices, model, figure=False, title=''):

    """
    substrings is a list of lists that contains substrings which are to be grouped together

    """

    colors = ['tab:blue','tab:red','tab:green','tab:purple','tab:orange']

    if feature_indices is None:
        feature_indices = list(range(5))

    outcome_collected = list()
    predictors_collected = list()

    for collection in substrings:

        outcome = list()
        predictors = list()
        for substring in collection:

            #dropping the name field here
            outcome.append(model_aucs[model_aucs['name'].str.contains(substring)].iloc[:,1:])
            predictors.append(feature_attributes[feature_attributes['name'].str.contains(substring)].iloc[:,1:])

        outcome = pd.concat(outcome, axis = 0)
        predictors = pd.concat(predictors, axis = 0)

        outcome_collected.append(outcome)
        predictors_collected.append(predictors)

    for i in range(len(substring_names)):

        X = predictors_collected[i].iloc[:,feature_indices]
        Y = outcome_collected[i]['test'].tolist()

        model.fit(X,Y)

        if figure and len(feature_indices) == 1:

            plt.scatter(X.iloc[:,0].tolist(), Y, c = colors[i])

            xs = np.linspace(min(X.iloc[:,0]) * 0.9, max(X.iloc[:,0]) * 1.1, 1000).reshape(-1,1)
            ys = model.predict(xs)

            plt.plot(xs, ys, c = colors[i], label = substring_names[i])

            
        print(f'{substring_names[i]}\n')
        for j in range(X.shape[1]):

            print(f'{X.columns[j]}: {round(model.coef_[j],2)}\n')

    if figure and len(feature_indices) == 1:
        plt.xlabel(X.columns[0])
        plt.legend()
        plt.ylabel('AUC')
        plt.title(title)
        plt.show()

