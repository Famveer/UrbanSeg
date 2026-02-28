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
        ml_task='classification',
        kernel_background_samples=100,   # kept for API parity with SHAPAnalyzer
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.ml_task = 'classification' if "class" in ml_task.lower() else 'regression'

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
            mode=self.ml_task,
            discretize_continuous=discretize_continuous,
            discretizer=discretizer,
            random_state=42,
            **kwargs,
        )

        return self

    def explain(
        self,
        X,
        sample_idx=None,
        num_features=10,
        num_samples=5000,
        labels_to_explain=1,
        **kwargs,
    ):
        """
        Explain one or multiple samples depending on sample_idx.

        Parameters
        ----------
        X : array-like or DataFrame
        sample_idx : int, list of int, or None
            - int          → explain that single sample, return one Explanation
            - list of int  → explain those samples, return list of Explanation
            - None         → explain ALL samples, return list of Explanation
        num_features : int — max features per explanation
        num_samples  : int — LIME perturbation samples (higher = slower, more accurate)
        labels_to_explain   : int — how many predicted classes to explain
        **kwargs
            Forwarded to explainer.explain_instance().

        Returns
        -------
        lime.explanation.Explanation          if sample_idx is int
        list of lime.explanation.Explanation  if sample_idx is list or None
        """
        if self.explainer is None:
            raise RuntimeError("Call fit(X_train) before explaining samples.")

        X_arr = self._transform_X(X)

        # ── Single sample ────────────────────────────────────────────────
        if isinstance(sample_idx, int):
            exp = self.explainer.explain_instance(
                data_row=X_arr[sample_idx],
                predict_fn=self._predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=labels_to_explain,
                **kwargs,
            )
            return exp

        # ── Multiple samples ─────────────────────────────────────────────
        indices = sample_idx if sample_idx is not None else range(len(X_arr))
        explanations = []
        for i in indices:
            exp = self.explainer.explain_instance(
                data_row=X_arr[i],
                predict_fn=self._predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=labels_to_explain,
                **kwargs,
            )
            explanations.append(exp)
            print(f"  Explained sample {i}")

        return explanations

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_explanation(self, 
                         exp, 
                         label=None, 
                         top_n_features=None, 
                         show=True, 
                         # Single parameters
                         figsize_single=(10, 5),
                         # Multi parameters
                         sample_indices=None,
                         ncols=2,
                         figsize_per_plot=(8, 4),
                         ):
        if not isinstance(exp, list):
            self.plot_single(exp, 
                             label=label, 
                             top_n_features=top_n_features, 
                             show=show, 
                             figsize=figsize_single)
            return
        
        self.plot_multiple(exp, 
                           sample_indices=sample_indices, 
                           label=label, 
                           top_n_features=top_n_features, 
                           show=show, 
                           ncols=ncols, 
                           figsize_per_plot=figsize_per_plot)

    def plot_single(self, exp, label=None, top_n_features=None, show=True, figsize=(10, 5)):
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
        pairs = exp.as_list(label=label)
        if top_n_features is not None:
            pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n_features]
        features, weights = zip(*pairs)

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
        else:
            plt.close(fig)
            
    def plot_multiple(
        self,
        explanations,
        sample_indices=None,
        label=None,
        top_n_features=None,
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
            pairs = exp.as_list(label=lbl)
            if top_n_features is not None:
                pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n_features]
            features, weights = zip(*pairs)
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
        else:
            plt.close(fig)
            
    def feature_importance_plot(self, explanations, label=None, top_n_features=15,
                               show=True, figsize=(10, 6)):
        """
        Aggregate LIME weights across multiple explanations to produce a
        pseudo-global feature importance chart (mean absolute LIME weight).

        Parameters
        ----------
        explanations : list of lime.explanation.Explanation
        label : int or None — class index (defaults to top label of first exp)
        top_n_features : int — features to display
        show : bool
        figsize : tuple

        Returns
        -------
        pd.DataFrame with columns ['feature', 'mean_abs_weight']
        """
        importance = self.top_features(explanations, label, top_n_features)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(importance))
        ax.barh(y_pos, importance['mean_abs_weight'], color='#2166ac', edgecolor='white', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance['feature'], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |LIME weight|", fontsize=11)
        ax.set_title("Pseudo-Global Feature Importance (LIME)", fontsize=13, fontweight='bold')
        
        # Annotate bars
        for i, v in enumerate(importance['mean_abs_weight'].values):
            ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=8, color='#2166ac')

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)


    def contribution_plot(self, explanations, label=None, top_n_features=15,
                          show=True, figsize=(10, 7)):
        """
        Bar chart showing mean positive (red) and mean negative (blue)
        LIME contributions per feature — same color scheme as waterfall / SHAP.

        Aggregates weights across multiple explanations:
          - Red  (right) → mean of positive LIME weights across samples
          - Blue (left)  → mean magnitude of negative LIME weights across samples

        Features are sorted by total mean |weight| (most impactful on top).

        Parameters
        ----------
        explanations : list of lime.explanation.Explanation
            From explain_multiple() or a list of explain_sample() results.
        label : int or None
            Class index to aggregate (defaults to top label of first explanation).
        top_n_features : int
            Max number of features to display.
        show : bool
        figsize : tuple

        Returns
        -------
        pd.DataFrame  columns=['feature', 'mean_pos', 'mean_neg', 'total']
        """
        lbl = label if label is not None else explanations[0].top_labels[0]

        # Accumulate positive and negative weights separately per base feature
        pos_accum   = {}
        neg_accum   = {}
        pos_count   = {}
        neg_count   = {}

        for exp in explanations:
            for feat, weight in exp.as_list(label=lbl):
                base = _strip_discretized_suffix(feat)
                if weight >= 0:
                    pos_accum[base] = pos_accum.get(base, 0.0) + weight
                    pos_count[base] = pos_count.get(base, 0) + 1
                else:
                    neg_accum[base] = neg_accum.get(base, 0.0) + weight   # stays negative
                    neg_count[base] = neg_count.get(base, 0) + 1

        # All features seen across both dicts
        all_features = set(pos_accum) | set(neg_accum)

        rows = []
        for feat in all_features:
            mean_pos =  pos_accum.get(feat, 0.0) / max(pos_count.get(feat, 1), 1)
            mean_neg = -neg_accum.get(feat, 0.0) / max(neg_count.get(feat, 1), 1)  # magnitude
            rows.append({
                'feature' : feat,
                'mean_pos': mean_pos,
                'mean_neg': mean_neg,
                'total'   : mean_pos + mean_neg,
            })

        df = (pd.DataFrame(rows)
                .sort_values('total', ascending=False)
                .head(top_n_features)
                .reset_index(drop=True))

        # Reverse so most important is on top in the horizontal bar chart
        df_plot  = df.iloc[::-1].reset_index(drop=True)
        y_pos    = np.arange(len(df_plot))

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(y_pos,  df_plot['mean_pos'], color='#d73027', edgecolor='white',
                height=0.6, label='Positive contribution')
        ax.barh(y_pos, -df_plot['mean_neg'], color='#4575b4', edgecolor='white',
                height=0.6, label='Negative contribution')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['feature'], fontsize=10)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Mean LIME weight', fontsize=11)

        class_label = (
            self.class_names[lbl]
            if self.class_names and lbl < len(self.class_names)
            else f'class {lbl}'
        )
        ax.set_title(
            f'Feature Contributions  (red = positive  |  blue = negative)  —  {class_label}',
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=10)

        # Annotate bar tips
        for i, row in df_plot.iterrows():
            if row['mean_pos'] > 0:
                ax.text(row['mean_pos'] + 0.0005, i,
                        f"+{row['mean_pos']:.4f}", va='center', fontsize=8, color='#d73027')
            if row['mean_neg'] > 0:
                ax.text(-row['mean_neg'] - 0.0005, i,
                        f"-{row['mean_neg']:.4f}", va='center', ha='right', fontsize=8, color='#4575b4')

        plt.tight_layout()
        if show:
            plt.show()
            return
        else:
            plt.close(fig)

        return df

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

    def get_lime_df(self, explanations, label=None, sample_indices=None):
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

    def get_feature_importance(self, explanations, label=None):
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
        ]).sort_values('mean_abs_weight', ascending=False).reset_index(drop=True)
        
        return importance

    def top_features(self, explanations, label=None, top_n_features=15):
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
        df = self.get_feature_importance(explanations, label=label).head(top_n_features)
        print(f"\nTop {top_n_features} features by mean |LIME weight|:")
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
