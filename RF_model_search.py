from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def checktraintestreg(X_train, X_test, y_train, y_test, test_model, ntrials =20):

    scores_train = np.zeros(ntrials)
    scores_test = np.zeros(ntrials)

    for i in range(ntrials):
        test_model.fit(X_train, y_train)
        pred_test = test_model.predict(X_test)
        pred_train = test_model.predict(X_train)

        scores_test[i] = (metrics.r2_score(y_test,pred_test))
        scores_train[i] =(metrics.r2_score(y_train,pred_train))

    print('Training scores '+str(scores_train.mean())+' +- '+str(scores_train.std()))
    print('Test scores '+str(scores_test.mean())+' +- '+str(scores_test.std()))


def RF_checktraintestreg_hyperparams(X_train, X_test, y_train, y_test, skip):
    
    '''
    This function iterates through different hyperparameter values,
    passing them to the checktraintest function and prints out the results.
    This helps in deciding the best model.
    '''
    n_features   = X_train.shape[1]
    num_trees    = [50, 100, 1000]
    min_leaf     = [1,5,10]
    depth        = [4,5,7,10,12,15]
    max_features = [3,5,n_features]
    
    if skip == False:
        print('*************************')
        print('FIND NUM TREES AND LEAF SAMPLES')
        print('*************************')
    
        for leaf in min_leaf:
            for trees in num_trees:
                for feats in max_features:
                    print('num_trees:', trees, ' min_leaf:',leaf, 'max_features:',feat)
                    checktraintestreg(X_train, X_test, y_train,\
                                      y_test,RandomForestRegressor(n_estimators=trees,\
                                                                   min_samples_leaf=leaf,\
                                                                   max_features=feats))
                    print('#########################')
                    print('#########################')
                
        print('*************************')
        print('FIND MAX DEPTH')
        print('*************************') 
        best_num_trees    = int(input("Enter number of trees: "))
        best_min_leaf     = int(input('Enter number of min_samples_leaf: '))
        best_max_features = int(input('Enter number of max_features: '))

        for num in depth: 
            print('num_trees:', best_num_trees,  'min_leaf:',best_min_leaf,\
                  'max_features:',best_max_features, 'max_depth:',num)
            checktraintestreg(X_train, X_test, y_train, y_test,\
                              RandomForestRegressor(n_estimators=best_num_trees,\
                                                min_samples_leaf=best_min_leaf,\
                                                max_features=best_max_features,\
                                                    max_depth=num))
            print('#########################')
            print('#########################')                
                
    
    else:
        print('*************************')
        print('FIND MAX DEPTH')
        print('*************************') 
        best_num_trees    = int(input("Enter number of trees: "))
        best_min_leaf     = int(input('Enter number of min_samples_leaf: '))
        best_max_features = int(input('Enter number of max_features: '))
        
        for num in depth:
            print('num_trees:', best_num_trees,  'min_leaf:',best_min_leaf, ' max_depth:',num)
            checktraintestreg(X_train, X_test, y_train, y_test,\
                          RandomForestRegressor(n_estimators=best_num_trees,\
                                                min_samples_leaf=best_min_leaf,\
                                                max_features=best_max_features,\
                                                max_depth=num))
             
            print('#########################')
            print('#########################')

            
def evaluate(model, rf_model, test_features, test_labels):
    '''This function returns scores for either a regression 
    or multi-class classification problem'''
    
    predictions = model.predict(test_features)
    if rf_model =='Reg':
        r2  = metrics.r2_score(test_labels, predictions)
        print('Test R2 score {:0.3f}'.format(r2))
        return r2
    elif rf_model=='Class':
        prec = metrics.precision_score(test_labels, predictions, average='weighted')
        acc = metrics.balanced_accuracy_score(test_labels, predictions)
        print('Test weighted precision score:', prec)
        print('Test balanced accuracy score:', acc)
        return prec
    

def Grid_Search(X_train, X_test, y_train, y_test, rf_model, skip):
    '''this function iterates over hyperparameters using sklearn's GridSearchCV'''
    #typical scoring: 
    #for regression: ['r2'], 
    #for classification: metrics.precision_score(y_true, y_pred, average=‘weighted’)
    if rf_model == 'Reg':
        rf = RandomForestRegressor()
        scores = ['r2']
    elif rf_model == 'Class':
        rf = RandomForestClassifier()
        scores = make_scorer(metrics.precision_score, average='weighted')
        #scores = {'balanced_accuracy': make_scorer(metrics.balanced_accuracy_score),
                  #'precision': make_scorer(metrics.precision_score)}
    
    n_features   = X_train.shape[1]
    
    if skip == False:
        if n_features>5:
            param_grid = {
                'bootstrap': [True],
                'n_estimators': [50, 100, 1000],
                'min_samples_leaf': [1, 5, 10],
                'max_features': [3, 5, n_features],
                'max_depth': [5,7,10,12,15],
                'min_samples_split': [2, 5, 10]
            }
        elif n_features<=5:
            param_grid = {
                'bootstrap': [True],
                'n_estimators': [50, 100, 1000],
                'min_samples_leaf': [1, 5, 10],
                'max_depth': [5,7,10,12,15],
                'min_samples_split': [2, 5, 10]
            }
        
        print('*************************')
        print('HYPERPARAMS SEARCH')
        print('*************************')
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, scoring=scores,\
                                   cv = 5, n_jobs = -1, verbose = 2,  return_train_score=True)
        
        grid_search.fit(X_train, y_train)

        train_score = np.mean(grid_search.cv_results_['mean_train_score'])
        train_std   = np.mean(grid_search.cv_results_['std_train_score'])
        best_grid   = grid_search.best_estimator_
        print('#########################')
        print('SEARCH RESULTS')
        print('#########################')
        print('Training scores: {:.3f}'.format(train_score), '+/-{:.3f}'.format(train_std))
        evaluate(best_grid, rf_model, X_test, y_test)
        
        print(grid_search.best_params_)

        n_trees   = grid_search.best_params_['n_estimators']
        min_leaf  = grid_search.best_params_['min_samples_leaf']
        max_dep   = grid_search.best_params_['max_depth']
        min_split = grid_search.best_params_['min_samples_split']
        
        if n_features>5:
            max_feat  = grid_search.best_params_['max_features']
            return n_trees, min_leaf, max_feat, max_dep, min_split
        else:
            return n_trees, min_leaf, max_dep, min_split
    
    else:
        if n_features>5:
            param_grid = {
                'bootstrap': [True],
                'n_estimators': [100],
                'min_samples_leaf': [5],
                'max_features': [3, 5, n_features],
                'max_depth': [5,7,10,12,15],
            }
            
        elif n_features<5:
            param_grid = {
                'bootstrap': [True],
                'n_estimators': [100],
                'min_samples_leaf': [5],
                'max_depth': [5,7,10,12,15],
            }
        
        print('*************************')
        print('SEARCHING FOR MAX FEATURES AND MAX DEPTH')
        print('*************************')
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, scoring=scores,\
                                   cv = 5, n_jobs = -1, verbose = 2,  return_train_score=True)
        
        grid_search.fit(X_train, y_train)

        train_score = np.mean(grid_search.cv_results_['mean_train_score'])
        train_std   = np.mean(grid_search.cv_results_['std_train_score'])
        best_grid   = grid_search.best_estimator_
        print('#########################')
        print('SEARCH RESULTS')
        print('#########################')
        print('Training score: {:.3f}'.format(train_score), '+/-{:.3f}'.format(train_std))
        evaluate(best_grid, rf_model, X_test, y_test)

        print(grid_search.best_params_)
        
        max_feat  = grid_search.best_params_['max_features']
        if n_features>5:
            max_dep   = grid_search.best_params_['max_depth']
            return max_feat, max_dep
        else:
            return max_dep
        