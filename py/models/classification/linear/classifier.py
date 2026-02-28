from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd

import pickle
import joblib
from pathlib import Path

class LinearClassifier():
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models_config = self._define_models()
        self.scaler_config = self._define_scalers()
        self.grid_single_searches = {}
        self.results = {}
        self.best_model = None
    
    def model_zoo(self):
        print( "Model zoo:", list( self.models_config.keys() ), "\n" )
    
    def _define_scalers(self):
        return {
                "standard": StandardScaler(),
                "empty": None,
               }
    
    def _define_models(self):
        """Define all models and their parameter grids."""
        return {
            'logistic_regression': {
                'model': LogisticRegression(
                    tol=1e-3,
                    dual=False,
                    random_state=self.random_state,
                    max_iter=1000,
                ),
                'params': {
                    'classifier__C': np.logspace(-2, 2, num=5),
                    'classifier__class_weight': [None, 'balanced'],
                    'classifier__penalty': ['none', 'l2'],
                }
            },
            'ridge': {
                'model': RidgeClassifier(
                    tol=1e-3,
                    random_state=self.random_state,
                    max_iter=1000,
                ),
                'params': {
                    'classifier__alpha': np.logspace(-2, 2, num=5),
                    'classifier__class_weight': [None, 'balanced'],
                }
            },
            'linear_svc': {
                'model': LinearSVC(
                    tol=1e-3,
                    random_state=self.random_state,
                    max_iter=1000,
                ),
                'params': {
                    'classifier__C': np.logspace(-2, 2, num=5),
                    'classifier__class_weight': [None, 'balanced'],
                    'classifier__penalty': ['l2'],
                    'classifier__loss': ['hinge', 'squared_hinge'],
                }
            },
            'svm': {
                'model': SVC(
                    tol=1e-3,
                    random_state=self.random_state,
                    max_iter=1000,
                    probability=True,
                ),
                'params': {
                    'classifier__C': np.logspace(-2, 2, num=5),
                    'classifier__class_weight': [None, 'balanced'],
                    'classifier__gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001],
                    'classifier__kernel': ['linear', 'rbf']
                }
            }
        }
    
    def add_model(self, name, model, params):
        """Add a new model configuration."""
        self.models_config[name] = {
            'model': model,
            'params': params
        }
    
    def remove_model(self, name):
        """Remove a model configuration."""
        if name in self.models_config:
            del self.models_config[name]
            
    def create_grid_single(self, 
                           scaler="standard", 
                           n_jobs=-1, 
                           cv_splits=5, 
                           scoring='balanced_accuracy'
                           ):
        """Create GridSearchCV objects for all configured models."""
        for name, config in self.models_config.items():
            pipeline = Pipeline(
                steps=[
                    ('scaler', self.scaler_config[scaler]),
                    ('classifier', config['model']),
                ],
            )
            
            self.grid_single_searches[name] = GridSearchCV(
                estimator=pipeline,
                param_grid=config['params'],
                scoring=scoring,
                n_jobs=n_jobs,
                refit=True,
                cv=StratifiedKFold(n_splits=cv_splits),
                verbose=1,
            )
        
        return self.grid_single_searches
    
    
    def fit_all(self, X_train, y_train):
        """Fit all models with GridSearch."""
        if not self.grid_single_searches:
            self.create_grid_single()
        
        for name, grid_search in self.grid_single_searches.items():
            print(f"\n{'='*60}")
            print(f"Fitting {name}...")
            print(f"{'='*60}")
            
            grid_search.fit(X_train, y_train)
            
            self.results[name] = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_,
                'grid_search': grid_search
            }
        
        # Find the best model overall
        self.best_model = max(self.results.items(), key=lambda x: x[1]['best_score'])
        
        return self.results
    
    
    def fit_single(self, model_name, X_train, y_train):
        """Fit a single model."""
        if not self.grid_single_searches:
            self.create_grid_single()
        
        if model_name not in self.grid_single_searches:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.grid_single_searches.keys())}")
        
        print(f"Fitting {model_name}...")
        grid_search = self.grid_single_searches[model_name]
        grid_search.fit(X_train, y_train)
        
        self.results[model_name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_,
            'grid_search': grid_search
        }
        
        return self.results[model_name]
            
    def get_results_summary(self):
        """Get a summary of all results as a DataFrame."""
        if not self.results:
            print("No results available. Run fit_all() or fit_single() first.")
            return None
        
        summary = []
        for name, result in self.results.items():
            summary.append({
                'model': name,
                'best_score': result['best_score'],
                'best_params': str(result['best_params'])
            })
        
        df = pd.DataFrame(summary).sort_values('best_score', ascending=False)
        return df
    
    def print_results(self):
        """Print detailed results for all models."""
        if not self.results:
            print("No results available. Run fit_all() or fit_single() first.")
            return
        
        print("\n" + "="*80)
        print("GRID SEARCH RESULTS SUMMARY")
        print("="*80)
        
        for name, result in sorted(self.results.items(), key=lambda x: x[1]['best_score'], reverse=True):
            print(f"\n{name.upper()}:")
            print(f"  Best CV Score: {result['best_score']:.4f}")
            print(f"  Best Parameters:")
            for param, value in result['best_params'].items():
                print(f"    {param}: {value}")
        
        if self.best_model:
            print(f"\n{'='*80}")
            print(f"BEST OVERALL MODEL: {self.best_model[0].upper()}")
            print(f"Best Score: {self.best_model[1]['best_score']:.4f}")
            print(f"{'='*80}")
    
    def get_best_model_name(self):
        """Return the best performing model."""
        if not self.best_model:
            print("No results available. Run fit_all() first.")
            return None
        
        return self.best_model[0]
    
    def get_best_model(self):
        """Return the best performing model."""
        if not self.best_model:
            print("No results available. Run fit_all() first.")
            return None
        
        return self.best_model[1]['best_estimator']
    
    def predict(self, X_test, model_name=None):
        """Make predictions using the best model or a specific model."""
        if model_name:
            if model_name not in self.results:
                raise ValueError(f"Model '{model_name}' not found in results.")
            return self.results[model_name]['best_estimator'].predict(X_test)
        else:
            if not self.best_model:
                raise ValueError("No best model available. Run fit_all() first.")
            return self.best_model[1]['best_estimator'].predict(X_test)
            
    
    def save(self, filepath, method='pickle'):
        """
        Save the entire instance to disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path where to save the instance
        method : str, default='joblib'
            Serialization method: 'joblib' (recommended) or 'pickle'
        
        Examples:
        ---------
        >>> model_search.save('models/my_search.pkl')
        >>> model_search.save('models/my_search.pkl', method='pickle')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if method == 'joblib':
            joblib.dump(self, filepath)
            print(f"Instance saved to {filepath} using joblib")
        elif method == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"Instance saved to {filepath} using pickle")
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'joblib' or 'pickle'")
    
    @classmethod
    def load(cls, filepath, method='pickle'):
        """
        Load an instance from disk.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the saved instance
        method : str, default='joblib'
            Serialization method: 'joblib' (recommended) or 'pickle'
        
        Returns:
        --------
        ModelGridSearch instance
        
        Examples:
        ---------
        >>> model_search = ModelGridSearch.load('models/my_search.pkl')
        >>> model_search = ModelGridSearch.load('models/my_search.pkl', method='pickle')
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if method == 'joblib':
            instance = joblib.load(filepath)
            print(f"Instance loaded from {filepath} using joblib")
        elif method == 'pickle':
            with open(filepath, 'rb') as f:
                instance = pickle.load(f)
            print(f"Instance loaded from {filepath} using pickle")
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'joblib' or 'pickle'")
        
        return instance
    
    def save_best_model_only(self, filepath, method='pickle'):
        """
        Save only the best model (not the entire search instance).
        This is useful for deployment when you only need the trained model.
        
        Parameters:
        -----------
        filepath : str or Path
            Path where to save the best model
        method : str, default='joblib'
            Serialization method: 'joblib' (recommended) or 'pickle'
        """
        if not self.best_model:
            raise ValueError("No best model available. Run fit_all() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        best_estimator = self.best_model[1]['best_estimator']
        
        if method == 'joblib':
            joblib.dump(best_estimator, filepath)
        elif method == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(best_estimator, f)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'joblib' or 'pickle'")
        
        print(f"Best model ({self.best_model[0]}) saved to {filepath}")
    
    @staticmethod
    def load_model(filepath, method='pickle'):
        """
        Load a saved model (not the entire search instance).
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the saved model
        method : str, default='joblib'
            Serialization method: 'joblib' (recommended) or 'pickle'
        
        Returns:
        --------
        Trained model (Pipeline)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if method == 'joblib':
            model = joblib.load(filepath)
        elif method == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'joblib' or 'pickle'")
        
        print(f"Model loaded from {filepath}")
        return model
