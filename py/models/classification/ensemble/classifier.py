from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path


class EnsembleClassifier():
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models_config = self._define_models()
        self.scaler_config = self._define_scalers()
        self.grid_single_searches = {}
        self.results = {}
        self.best_model = None

    def model_zoo(self):
        print("Model zoo:", list(self.models_config.keys()), "\n")

    def _define_scalers(self):
        return {
            "standard": StandardScaler(),
            "empty": None,
        }

    def _define_models(self):
        """Define all ensemble models and their parameter grids."""
        return {
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'classifier__max_features': [None, 'sqrt'],
                    'classifier__max_depth': np.append(None, np.arange(10, 80, 10)),
                    'classifier__min_samples_split': np.arange(3, 7),
                    'classifier__min_samples_leaf': np.arange(3, 7),
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'classifier__n_estimators': [200],
                    'classifier__bootstrap': [True],
                    'classifier__max_features': [None, 'sqrt'],
                    'classifier__max_depth': np.append(None, np.arange(10, 80, 10)),
                    'classifier__min_samples_split': np.arange(3, 7),
                    'classifier__min_samples_leaf': np.arange(3, 7),
                    'classifier__class_weight': [None, 'balanced'],
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'classifier__n_estimators': [100],
                    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'classifier__subsample': [0.8, 1.0],
                    'classifier__max_features': [None, 'sqrt'],
                    'classifier__max_depth': [3, 5, 10],
                    'classifier__min_samples_split': np.arange(3, 7),
                    'classifier__min_samples_leaf': np.arange(3, 7),
                }
            },
            'adaboost': {
                'model': AdaBoostClassifier(random_state=self.random_state),
                'params': {
                    'classifier__n_estimators': np.arange(100, 600, 100),
                    'classifier__learning_rate': np.round(np.arange(0.1, 1.1, 0.2), 1),
                }
            },
            # ----------------------------------------------------------------
            # Extra Trees — like Random Forest but uses random thresholds for
            # each feature, making it faster and often less prone to overfitting.
            # No extra packages needed (sklearn.ensemble).
            # ----------------------------------------------------------------
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=self.random_state),
                'params': {
                    'classifier__n_estimators': [200],
                    'classifier__max_features': [None, 'sqrt'],
                    'classifier__max_depth': np.append(None, np.arange(10, 80, 10)),
                    'classifier__min_samples_split': np.arange(3, 7),
                    'classifier__min_samples_leaf': np.arange(3, 7),
                    'classifier__class_weight': [None, 'balanced'],
                    'classifier__bootstrap': [True, False],
                }
            },
            # ----------------------------------------------------------------
            # Histogram-based Gradient Boosting — much faster than standard
            # GradientBoosting on large datasets (similar to LightGBM/XGBoost).
            # No extra packages needed (sklearn.ensemble).
            # ----------------------------------------------------------------
            'hist_gradient_boosting': {
                'model': HistGradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'classifier__max_iter': [200],
                    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'classifier__max_depth': [None, 5, 10, 20],
                    'classifier__min_samples_leaf': np.arange(10, 40, 10),
                    'classifier__l2_regularization': [0.0, 0.1, 1.0],
                    'classifier__class_weight': [None, 'balanced'],
                }
            },
            # ----------------------------------------------------------------
            # Bagging — trains N copies of a base estimator on random subsets
            # of the data, reducing variance. Works well with any base learner.
            # No extra packages needed (sklearn.ensemble).
            # ----------------------------------------------------------------
            'bagging': {
                'model': BaggingClassifier(
                    estimator=DecisionTreeClassifier(),
                    random_state=self.random_state
                ),
                'params': {
                    'classifier__n_estimators': [200],
                    'classifier__max_samples': [0.6, 0.8, 1.0],
                    'classifier__max_features': [0.6, 0.8, 1.0],
                    'classifier__bootstrap': [True, False],
                    'classifier__bootstrap_features': [True, False],
                }
            },
            # ----------------------------------------------------------------
            # XGBoost — gradient boosting with regularization (L1/L2),
            # column subsampling, and highly optimized tree building.
            # pip install xgboost
            # ----------------------------------------------------------------
            'xgboost': {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    verbosity=0,
                    use_label_encoder=False,
                ),
                'params': {
                    'classifier__n_estimators': [200],
                    'classifier__learning_rate': [0.05],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__subsample': [0.7, 0.9, 1.0],
                    'classifier__colsample_bytree': [0.7, 0.9, 1.0],
                    'classifier__gamma': [0, 0.1, 0.5],
                    'classifier__reg_alpha': [0.1],       # L1
                    'classifier__reg_lambda': [0.5],    # L2
                    'classifier__scale_pos_weight': [1, 5, 10],   # class imbalance
                }
            },
            # ----------------------------------------------------------------
            # LightGBM — leaf-wise tree growth, very fast on large datasets,
            # low memory footprint, native support for categorical features.
            # pip install lightgbm
            # ----------------------------------------------------------------
            'lightgbm': {
                'model': LGBMClassifier(
                    random_state=self.random_state,
                    verbosity=-1,
                ),
                'params': {
                    'classifier__n_estimators': [200],
                    'classifier__learning_rate': [0.05],
                    'classifier__num_leaves': [31, 63, 127],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__subsample': [0.7, 0.9, 1.0],
                    'classifier__colsample_bytree': [0.7, 0.9, 1.0],
                    'classifier__reg_alpha': [0.1],       # L1
                    'classifier__reg_lambda': [0.1],      # L2
                    'classifier__min_child_samples': [10, 20, 50],
                    'classifier__class_weight': [None, 'balanced'],
                }
            },
            # ----------------------------------------------------------------
            # CatBoost — gradient boosting with built-in ordered boosting and
            # native categorical feature support; often strong out-of-the-box.
            # pip install catboost
            # ----------------------------------------------------------------
            'catboost': {
                'model': CatBoostClassifier(
                    random_state=self.random_state,
                    verbose=0,
                ),
                'params': {
                    'classifier__iterations': [200],
                    'classifier__learning_rate': [0.05],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__l2_leaf_reg': [3, 5, 7],
                    'classifier__subsample': [0.7, 0.9, 1.0],
                    'classifier__colsample_bylevel': [0.7, 0.9, 1.0],
                    'classifier__auto_class_weights': [None, 'Balanced'],
                }
            },
        }

    def add_model(self, name, model, params):
        """Add a new model configuration."""
        self.models_config[name] = {'model': model, 'params': params}

    def remove_model(self, name):
        """Remove a model configuration."""
        if name in self.models_config:
            del self.models_config[name]

    def create_grid_single(self,
                           scaler="standard",
                           n_jobs=-1,
                           cv_splits=5,
                           scoring='balanced_accuracy'):
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

        self.best_model = max(self.results.items(), key=lambda x: x[1]['best_score'])
        return self.results

    def fit_single(self, model_name, X_train, y_train):
        """Fit a single model."""
        if not self.grid_single_searches:
            self.create_grid_single()

        if model_name not in self.grid_single_searches:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self.grid_single_searches.keys())}"
            )

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

        for name, result in sorted(self.results.items(),
                                   key=lambda x: x[1]['best_score'],
                                   reverse=True):
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
        """Return the name of the best performing model."""
        if not self.best_model:
            print("No results available. Run fit_all() first.")
            return None
        return self.best_model[0]

    def get_best_model(self):
        """Return the best performing estimator."""
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
        method : str — 'joblib' (recommended for sklearn) or 'pickle'
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
        """Load an instance from disk."""
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
        """Save only the best estimator (useful for deployment)."""
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
        """Load a saved estimator (not the full search instance)."""
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
