import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from joblib import dump, load

class Classicator:
    def __init__(self, data_train, data_test,target_name):
        self.X = data_train.drop(target_name,axis=1)
        self.y = data_train[target_name]
        self.X_test = data_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X,self.y, random_state=42)
        
        self.class_classicator = None
        self.simple_classificator = None
    
    def set_parametr_search(self, parametrs_search):
        self.parametrs_search = parametrs_search
    
    def set_class_searcher(self,class_searcher, cv, scoring, n_jobs):
        self.class_searcher = class_searcher
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        
    def set_class_classicator(self, class_classicator):
#       class_classicator - класс который будет использоваться для классификации 
        self.class_classicator = class_classicator
        
    def searh_best_classificator(self):
        clf = self.class_searcher(
            self.class_classicator(),
            param_grid=self.parametrs_search,
            cv=self.cv,
            scoring=self.scoring, 
            n_jobs=self.n_jobs
        )
        clf.fit(self.X, self.y)
        
        self.__print_report_searcher(clf)
        self.__print_report_classificator(clf)
        
        self.best_parametrs = clf.best_params_
        
    def make_best_classificator(self):
        model = self.class_classicator(**self.best_parametrs)
        model.fit(self.X_train, self.y_train)
        self.__print_report_classificator(model)
        self.best_classificator = model
    
    def make_simple_classificator(self):
        model = self.class_classicator()
        model.fit(self.X_train, self.y_train)
        self.__print_report_classificator(model)
        self.simple_classificator = model
    
    def make_final_classificator(self):
        model = self.class_classicator(**self.best_parametrs)
        model.fit(self.X, self.y)
        self.__print_report_classificator(model)
        self.final_classificator = model
    
    def make_predict_with_final_classificator(self):
        y_pred = self.final_classificator.predict(self.X_test)
        return y_pred
        
    def save_final_model(self, path_save, name):
        dump(self.final_classificator, path_save + name)
        
    def __print_report_searcher(self,model):
        print('Best estimator')
        print('')
        print(model.best_estimator_)
        print('')
        print('Best parametrs')
        print('')
        print(model.best_params_)
        print('')
    
    def __print_report_classificator(self, model):
        y_pred = model.predict(self.X_valid)
        print(self.class_classicator.__name__)
        print(classification_report(self.y_valid, y_pred))
        