import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

from lime import lime_tabular

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import label_binarize

# Optional third-party imports
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model sets for predict-function routing
# ---------------------------------------------------------------------------
_DECISION_FUNCTION_MODELS = (
    LinearSVC,
    RidgeClassifier,
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class LIMEAnalyzer:
    """
    Unified LIME analysis for models trained with LinearClassifier or
    EnsembleClassifier.

    LIME is inherently local — it explains one prediction at a time. Use
    `explain_sample()` for single explanations and `explain_multiple()` to
    batch several samples.

    Parameters
    ----------
    model : sklearn Pipeline or fitted estimator
        Typically `classifier_instance.get_best_model()` or
        `classifier_instance.results['model_name']['best_estimator']`.
    feature_names : list of str, optional
        Column names for X. If None, inferred from DataFrame or auto-generated.
    class_names : list of str, optional
        Class labels (e.g. ['benign', 'malignant']). Shown in plots.
    mode : str, default='classification'
        'classification' or 'regression'.
    kernel_background_samples : int, default=100
        Not used by LIME directly, kept for API consistency with SHAPAnalyzer.

    Examples
    --------
    >>> # From LinearClassifier
    >>> lc = LinearClassifier()
    >>> lc.fit_all(X_train, y_train)
    >>> analyzer = LIMEAnalyzer(lc.get_best_model(), feature_names=X.columns.tolist())
    >>> analyzer.fit(X_train)
    >>> exp = analyzer.explain_sample(X_test, sample_idx=0)
    >>> analyzer.plot_explanation(exp)

    >>> # From EnsembleClassifier
    >>> ec = EnsembleClassifier()
    >>> ec.fit_all(X_train, y_train)
    >>> analyzer = LIMEAnalyzer(
    ...     ec.results['xgboost']['best_estimator'],
    ...     feature_names=X.columns.tolist(),
    ...     class_names=['class_0', 'class_1']
    ... )
    >>> analyzer.fit(X_train)
    >>> analyzer.explain_sample(X_test, sample_idx=3, plot=True)
    """

    def __init__(
        self,
        model,
        feature_names=None,
        class_names=None,
        mode='classification',
        kernel_background_samples=100,   # kept for API parity with SHAPAnalyzer
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode

        self.explainer = None
        self._classifier = None
        self._scaler = None
        self._predict_fn = None
        self._predict_fn_type = None   # 'proba' | 'decision' | 'predict'
        self._n_classes = None

        self._unpack_pipeline()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_pipeline(self):
        """Extract scaler and classifier from a sklearn Pipeline."""
        if isinstance(self.model, Pipeline):
            steps = dict(self.model.steps)
            self._scaler = steps.get('scaler', None)
            self._classifier = steps.get('classifier', None)
            if self._classifier is None:
                self._classifier = self.model.steps[-1][1]
        else:
            self._classifier = self.model
            self._scaler = None

    def _transform_X(self, X):
        """Scale X if a scaler is present; always return np.ndarray."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)
        return X_arr

    def _resolve_feature_names(self, X):
        if self.feature_names is not None:
            return list(self.feature_names)
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        return [f"feature_{i}" for i in range(X.shape[1])]

    def _build_predict_fn(self, classes):
        """
        Build a predict function that always returns a (n_samples, n_classes)
        probability array — required by LimeTabularExplainer.

        Routing:
          1. predict_proba  — most classifiers
          2. decision_function + softmax — LinearSVC, RidgeClassifier
          3. predict + one-hot           — ultimate fallback
        """
        clf = self._classifier
        n_classes = len(classes)

        # --- Route 1: predict_proba via Pipeline ---
        if hasattr(self.model, 'predict_proba'):
            self._predict_fn_type = 'proba'
            def predict_fn(X_arr):
                return self.model.predict_proba(
                    self._inverse_transform(X_arr)
                )
            return predict_fn

        # --- Route 2: decision_function (LinearSVC, RidgeClassifier) ---
        if isinstance(clf, _DECISION_FUNCTION_MODELS) or hasattr(clf, 'decision_function'):
            self._predict_fn_type = 'decision'
            print(
                f"[LIMEAnalyzer] '{type(clf).__name__}' has no predict_proba. "
                "Using softmax(decision_function) as probability proxy."
            )
            def predict_fn(X_arr):
                raw = self.model.decision_function(
                    self._inverse_transform(X_arr)
                )
                # Binary case: decision_function returns (n,); expand to (n,2)
                if raw.ndim == 1:
                    raw = np.column_stack([-raw, raw])
                return _softmax(raw)
            return predict_fn

        # --- Route 3: predict + one-hot fallback ---
        self._predict_fn_type = 'predict'
        warnings.warn(
            f"[LIMEAnalyzer] '{type(clf).__name__}' has no predict_proba or "
            "decision_function. Falling back to one-hot encoded predict().",
            UserWarning,
        )
        def predict_fn(X_arr):
            preds = self.model.predict(self._inverse_transform(X_arr))
            return label_binarize(preds, classes=classes).astype(float)
        return predict_fn

    def _inverse_transform(self, X_arr):
        """
        LIME perturbs samples in the original feature space and passes them
        through the predict function. Since our Pipeline already has a scaler,
        we must NOT re-scale — just convert to the right dtype.
        """
        return X_arr.astype(np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train, categorical_features=None, discretize_continuous=True,
            discretizer='quartile', **kwargs):
        """
        Build the LimeTabularExplainer using training data.

        Parameters
        ----------
        X_train : array-like or DataFrame, shape (n_samples, n_features)
            Training data — used to compute feature statistics for perturbation.
        categorical_features : list of int, optional
            Column indices of categorical features (LIME treats them differently).
        discretize_continuous : bool, default=True
            Whether to discretize continuous features for the explanation.
        discretizer : str, default='quartile'
            'quartile', 'decile', or 'entropy'.
        **kwargs
            Extra arguments forwarded to LimeTabularExplainer.

        Returns
        -------
        self
        """
        self.feature_names = self._resolve_feature_names(X_train)
        X_train_arr = self._transform_X(X_train)

        # Infer classes from training labels is not possible here, so we rely
        # on class_names length or default binary
        classes = (
            list(range(len(self.class_names)))
            if self.class_names is not None
            else [0, 1]
        )
        self._n_classes = len(classes)

        self._predict_fn = self._build_predict_fn(classes)

        print(
            f"[LIMEAnalyzer] Building LimeTabularExplainer for "
            f"'{type(self._classifier).__name__}' "
            f"(predict_fn: {self._predict_fn_type})"
        )

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_arr,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=categorical_features,
            mode=self.mode,
            discretize_continuous=discretize_continuous,
            discretizer=discretizer,
            random_state=42,
            **kwargs,
        )

        return self

    def explain_sample(
        self,
        X,
        sample_idx=0,
        num_features=10,
        num_samples=5000,
        top_labels=1,
        plot=False,
        show=True,
        **kwargs,
    ):
        """
        Explain a single prediction.

        Parameters
        ----------
        X : array-like or DataFrame
        sample_idx : int — which row to explain
        num_features : int — max features in the explanation
        num_samples : int — perturbation samples for LIME
        top_labels : int — how many predicted classes to explain
        plot : bool — show the matplotlib plot immediately
        show : bool — call plt.show() (only used when plot=True)
        **kwargs
            Forwarded to explainer.explain_instance().

        Returns
        -------
        lime.explanation.Explanation
        """
        if self.explainer is None:
            raise RuntimeError("Call fit(X_train) before explaining samples.")

        X_arr = self._transform_X(X)
        instance = X_arr[sample_idx]

        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self._predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=top_labels,
            **kwargs,
        )

        if plot:
            self.plot_explanation(exp, show=show)

        return exp

    def explain_multiple(
        self,
        X,
        sample_indices=None,
        num_features=10,
        num_samples=5000,
        top_labels=1,
        **kwargs,
    ):
        """
        Explain multiple samples and return a list of Explanation objects.

        Parameters
        ----------
        X : array-like or DataFrame
        sample_indices : list of int or None — defaults to all samples
        num_features : int
        num_samples : int
        top_labels : int

        Returns
        -------
        list of lime.explanation.Explanation
        """
        if self.explainer is None:
            raise RuntimeError("Call fit(X_train) before explaining samples.")

        X_arr = self._transform_X(X)
        indices = sample_indices if sample_indices is not None else range(len(X_arr))

        explanations = []
        for i in indices:
            exp = self.explainer.explain_instance(
                data_row=X_arr[i],
                predict_fn=self._predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=top_labels,
                **kwargs,
            )
            explanations.append(exp)
            print(f"  Explained sample {i}")

        return explanations

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_explanation(self, exp, label=None, show=True, figsize=(10, 5)):
        """
        Matplotlib bar chart for a single LIME Explanation object.

        Parameters
        ----------
        exp : lime.explanation.Explanation
        label : int or None — class index to plot (defaults to top predicted)
        show : bool
        figsize : tuple
        """
        label = label if label is not None else exp.top_labels[0]
        features, weights = zip(*exp.as_list(label=label))

        colors = ['#d73027' if w > 0 else '#4575b4' for w in weights]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(features))
        ax.barh(y_pos, weights, color=colors, edgecolor='white', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("LIME weight", fontsize=11)

        class_label = (
            self.class_names[label]
            if self.class_names and label < len(self.class_names)
            else f"class {label}"
        )
        ax.set_title(f"LIME Explanation — {class_label}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_multiple(
        self,
        explanations,
        sample_indices=None,
        label=None,
        ncols=2,
        figsize_per_plot=(8, 4),
        show=True,
    ):
        """
        Grid of LIME bar charts for multiple explanations.

        Parameters
        ----------
        explanations : list of lime.explanation.Explanation
        sample_indices : list of int or None — used for subplot titles
        label : int or None — class index to plot
        ncols : int — number of columns in the grid
        figsize_per_plot : tuple
        show : bool
        """
        n = len(explanations)
        nrows = int(np.ceil(n / ncols))
        fig_w = figsize_per_plot[0] * ncols
        fig_h = figsize_per_plot[1] * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        axes = np.array(axes).flatten()

        for i, exp in enumerate(explanations):
            lbl = label if label is not None else exp.top_labels[0]
            features, weights = zip(*exp.as_list(label=lbl))
            colors = ['#d73027' if w > 0 else '#4575b4' for w in weights]
            y_pos = np.arange(len(features))

            axes[i].barh(y_pos, weights, color=colors, edgecolor='white', height=0.6)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(features, fontsize=8)
            axes[i].axvline(0, color='black', linewidth=0.8)

            idx_label = f"Sample {sample_indices[i]}" if sample_indices else f"Sample {i}"
            class_label = (
                self.class_names[lbl]
                if self.class_names and lbl < len(self.class_names)
                else f"class {lbl}"
            )
            axes[i].set_title(f"{idx_label} — {class_label}", fontsize=10, fontweight='bold')

        # Hide unused subplots
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_global_importance(self, explanations, label=None, top_n=15,
                               show=True, figsize=(10, 6)):
        """
        Aggregate LIME weights across multiple explanations to produce a
        pseudo-global feature importance chart (mean absolute LIME weight).

        Parameters
        ----------
        explanations : list of lime.explanation.Explanation
        label : int or None — class index (defaults to top label of first exp)
        top_n : int — features to display
        show : bool
        figsize : tuple

        Returns
        -------
        pd.DataFrame with columns ['feature', 'mean_abs_weight']
        """
        lbl = label if label is not None else explanations[0].top_labels[0]

        # Accumulate weights per feature (feature names may include discretized ranges)
        weight_accum = {}
        count_accum = {}
        for exp in explanations:
            for feat, weight in exp.as_list(label=lbl):
                # Normalise to base feature name for cleaner aggregation
                base = _strip_discretized_suffix(feat)
                weight_accum[base] = weight_accum.get(base, 0.0) + abs(weight)
                count_accum[base] = count_accum.get(base, 0) + 1

        importance = pd.DataFrame([
            {'feature': k, 'mean_abs_weight': weight_accum[k] / count_accum[k]}
            for k in weight_accum
        ]).sort_values('mean_abs_weight', ascending=False).head(top_n).reset_index(drop=True)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(importance))
        ax.barh(y_pos, importance['mean_abs_weight'], color='#2166ac', edgecolor='white', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance['feature'], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |LIME weight|", fontsize=11)
        ax.set_title("Pseudo-Global Feature Importance (LIME)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()

        return importance

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_explanation_df(self, exp, label=None):
        """
        Return a single explanation as a DataFrame.

        Parameters
        ----------
        exp : lime.explanation.Explanation
        label : int or None

        Returns
        -------
        pd.DataFrame with columns ['feature', 'weight']
        """
        lbl = label if label is not None else exp.top_labels[0]
        rows = exp.as_list(label=lbl)
        return pd.DataFrame(rows, columns=['feature', 'weight'])

    def get_explanations_df(self, explanations, label=None, sample_indices=None):
        """
        Stack multiple explanations into one long-format DataFrame.

        Parameters
        ----------
        explanations : list of lime.explanation.Explanation
        label : int or None
        sample_indices : list of int or None

        Returns
        -------
        pd.DataFrame with columns ['sample_idx', 'feature', 'weight']
        """
        dfs = []
        for i, exp in enumerate(explanations):
            df = self.get_explanation_df(exp, label=label)
            df.insert(0, 'sample_idx', sample_indices[i] if sample_indices else i)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def top_features(self, explanations, label=None, n=10):
        """
        Print and return the top-n features by mean |LIME weight| across
        multiple explanations.

        Parameters
        ----------
        explanations : list of lime.explanation.Explanation
        label : int or None
        n : int

        Returns
        -------
        pd.DataFrame
        """
        df = self.plot_global_importance(explanations, label=label, top_n=n, show=False)
        print(f"\nTop {n} features by mean |LIME weight|:")
        print(df.to_string(index=False))
        return df


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _softmax(x):
    """Row-wise softmax."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _strip_discretized_suffix(feature_str):
    """
    LIME discretizes continuous features into ranges like 'age > 45.00'.
    Strip the comparison operator to recover the base feature name.
    """
    for op in [' <= ', ' > ', ' < ', ' >= ', ' == ', ' != ']:
        if op in feature_str:
            return feature_str.split(op)[0].strip()
    return feature_str.strip()
