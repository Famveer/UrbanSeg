import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


class SHAPAnalyzer:
    """
    Parameters
    ----------
    model : fitted sklearn Pipeline
        e.g. lc.get_best_model() or ec.results['xgboost']['best_estimator']
    ml_type : str
        'linear' (from LinearClassifier) or 'ensemble' (from EnsembleClassifier)
    model_name : str
        Name of the model, e.g. 'svm', 'logistic_regression', 'xgboost'.
        Used to decide whether KernelExplainer is needed.
    feature_names : list of str, optional
    class_names : list of str, optional
    n_background : int, default=100
        Number of background samples for KernelExplainer / LinearExplainer.

    Examples
    --------
    >>> # LinearClassifier
    >>> lc = LinearClassifier()
    >>> lc.fit_all(X_train, y_train)
    >>> analyzer = SHAPAnalyzer(
    ...     model      = lc.get_best_model(),
    ...     ml_type    = 'linear',
    ...     model_name = lc.get_best_model_name(),
    ...     feature_names = X.columns.tolist(),
    ... )
    >>> analyzer.fit(X_train)
    >>> sv = analyzer.shap_values(X_test)

    >>> # EnsembleClassifier
    >>> ec = EnsembleClassifier()
    >>> ec.fit_all(X_train, y_train)
    >>> analyzer = SHAPAnalyzer(
    ...     model      = ec.get_best_model(),
    ...     ml_type    = 'ensemble',
    ...     model_name = ec.get_best_model_name(),
    ...     feature_names = X.columns.tolist(),
    ... )
    >>> analyzer.fit(X_train)
    >>> sv = analyzer.shap_values(X_test)
    """

    def __init__(
        self,
        model,
        ml_type,
        model_name,
        feature_names=None,
        class_names=None,
        n_background=100,
    ):
        self.model         = model
        self.ml_type       = ml_type.lower()
        self.model_name    = model_name.lower()
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.class_names   = class_names
        self.n_background  = n_background

        self.explainer      = None
        self.explainer_type = None   # 'kernel' | 'linear' | 'tree'
        self._background    = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_array(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.array(X)

    def _resolve_feature_names(self, X):
        if self.feature_names is not None:
            return self.feature_names
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        return [f"feature_{i}" for i in range(np.array(X).shape[1])]

    def _get_classifier(self):
        return self.model.named_steps.get('classifier', self.model.steps[-1][1])

    def _get_scaler(self):
        return self.model.named_steps.get('scaler', None)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X_train):
        """
        Build the SHAP explainer using training data.

        Routing logic (mirrors your working code):
          LinearClassifier + SVM  →  KernelExplainer
          LinearClassifier + rest →  LinearExplainer
          EnsembleClassifier      →  shap.Explainer (tree-aware)

        Parameters
        ----------
        X_train : array-like or DataFrame

        Returns
        -------
        self
        """
        self.feature_names = self._resolve_feature_names(X_train)
        X_arr = self._to_array(X_train)

        n_bg = min(self.n_background, X_arr.shape[0])
        self._background = shap.sample(X_arr, n_bg)

        if self.ml_type == 'linear' and self.model_name == 'svm':
            # SVM has no linear/tree structure — use KernelExplainer
            self.explainer_type = 'kernel'
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self._background,
            )

        elif self.ml_type == 'linear':
            # LinearExplainer works on the raw classifier, but needs scaled data
            self.explainer_type = 'linear'
            scaler = self._get_scaler()
            bg_scaled = scaler.transform(self._background) if scaler else self._background
            self.explainer = shap.LinearExplainer(
                self._get_classifier(),
                bg_scaled,
            )

        else:
            # All ensemble models: shap.Explainer auto-routes to TreeExplainer
            self.explainer_type = 'tree'
            self.explainer = shap.TreeExplainer(self._get_classifier())

        print(f"[SHAPAnalyzer] '{self.model_name}' → {self.explainer_type.upper()}Explainer")
        return self

    # ------------------------------------------------------------------
    # Compute SHAP values
    # ------------------------------------------------------------------

    def shap_values(self, X, check_additivity=False):
        """
        Compute SHAP values for X.

        Returns
        -------
        shap.Explanation  (feature_names always populated)
        """
        if self.explainer is None:
            raise RuntimeError("Call fit(X_train) before computing SHAP values.")

        X_arr = self._to_array(X)

        if self.explainer_type == 'linear':
            scaler = self._get_scaler()
            X_in = scaler.transform(X_arr) if scaler else X_arr
            sv = self.explainer(X_in)
        elif self.explainer_type == 'tree':
            sv = self.explainer(X_arr, check_additivity=check_additivity)
        else:  # kernel
            sv = self.explainer(X_arr)

        # Always inject feature names so all plot functions show real names
        sv.feature_names = self.feature_names
        return sv

    # ------------------------------------------------------------------
    # get_shap_dict  — exact mirror of your one-liner
    # ------------------------------------------------------------------

    def get_shap_dict(self, X, sample_idx=0):
        """
        Return {feature_name: shap_value} for one sample.

        Mirrors your working code:
            contributions = shap_explainer(instance)
            shap_values_dict = dict(zip(feature_names, contributions[0].values))

        Parameters
        ----------
        X : array-like or DataFrame
        sample_idx : int

        Returns
        -------
        dict
        """
        X_arr    = self._to_array(X)
        instance = X_arr[sample_idx].reshape(1, -1)
        contributions = self.shap_values(instance)
        return dict(zip(self.feature_names, contributions[0].values))

    # ------------------------------------------------------------------
    # Convenience extractors
    # ------------------------------------------------------------------

    def get_shap_df(self, contributions):
        """
        SHAP values as a DataFrame (n_samples × n_features).
        For multiclass outputs, class index 1 is used.
        """
        vals = contributions.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        return pd.DataFrame(vals, columns=self.feature_names)

    def get_feature_importance(self, contributions, sort=True):
        """
        Mean absolute SHAP per feature.

        Returns
        -------
        pd.DataFrame  columns=['feature', 'mean_abs_shap']
        """
        shap_df = self.get_shap_df(contributions)
        df = shap_df.abs().mean().reset_index()
        df.columns = ['feature', 'mean_abs_shap']
        if sort:
            df = df.sort_values('mean_abs_shap', ascending=False)
        return df.reset_index(drop=True)

    def top_features(self, contributions, top_n_features=15):
        """Print and return top-n features by mean |SHAP|."""
        df = self.get_feature_importance(contributions).head(top_n_features)
        print(f"\nTop {top_n_features} features by mean |SHAP|:")
        print(df.to_string(index=False))
        return df

    # ------------------------------------------------------------------
    # Plots — thin wrappers around native SHAP plot functions
    # ------------------------------------------------------------------

    def _is_multiclass(self, sv):
        """True when SHAP returns (n_samples, n_features, n_classes)."""
        return sv.values.ndim == 3
    
    def _slice_class(self, sv, class_idx):
        """
        Return a new Explanation object restricted to one class.
        Required by bar, beeswarm, waterfall which cannot handle 3-D arrays.
        """
        import copy
        sv2 = copy.copy(sv)
        sv2.values        = sv.values[:, :, class_idx]
        sv2.base_values   = sv.base_values[:, class_idx]
        sv2.feature_names = self.feature_names   # <-- keeps real names in plots
        # data stays the same (2-D)
        return sv2

    def summary_plot(self, X, sv, class_idx=1, plot_type='dot', max_display=20, show=True):
        """Beeswarm (dot) or bar summary plot."""
        X_arr = self._to_array(X)

        if self._is_multiclass(sv):
            sv_plot = self._slice_class(sv, class_idx)
        else:
            sv_plot = sv
        
        shap.summary_plot(
            sv_plot.values, X_arr,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            class_names=self.class_names,
            show=show,
        )
        
    def dependence_plot(self, feature, X, sv, class_idx=1, interaction_feature='auto', show=True):
        """Dependence plot for a single feature."""
        X_arr = self._to_array(X)
        vals  = sv.values

        if self._is_multiclass(sv):
            vals = vals[:, :, class_idx]

        shap.dependence_plot(
            feature, vals, X_arr,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=show,
        )

    def bar_plot(self, sv, class_idx=1, max_display=10, show=True):
        """Global mean |SHAP| bar chart."""
        if self._is_multiclass(sv):
            sv = self._slice_class(sv, class_idx)

        shap.plots.bar(sv, max_display=max_display, show=show)

    def beeswarm_plot(self, sv, class_idx=1, max_display=10, show=True):
        """Beeswarm plot."""
        if self._is_multiclass(sv):
            sv = self._slice_class(sv, class_idx)

        shap.plots.beeswarm(sv, max_display=max_display, show=show)
        
    def contribution_plot(self, sv, class_idx=1, max_display=10, show=True, figsize=(10, 7)):
        """
        Bar chart showing mean positive (red) and mean negative (blue)
        SHAP contributions per feature — same color scheme as waterfall.

        Each feature gets two bars stacked from zero:
          - Red  → average of positive SHAP values across samples
          - Blue → average of negative SHAP values across samples

        Features are sorted by total mean |SHAP| (most impactful on top).

        Parameters
        ----------
        X           : array-like or DataFrame
        class_idx   : int   — class to analyse for multiclass models (default=1)
        max_display : int   — max number of features to show
        show        : bool
        figsize     : tuple
        """
        if self._is_multiclass(sv):
            sv = self._slice_class(sv, class_idx)

        vals = sv.values   # (n_samples, n_features)

        # Separate positive and negative contributions per feature
        pos_mean =  np.where(vals > 0, vals, 0).mean(axis=0)
        neg_mean = -np.where(vals < 0, vals, 0).mean(axis=0)   # stored as positive magnitude

        # Sort by total importance and trim to max_display
        order      = np.argsort(pos_mean + neg_mean)[::-1][:max_display]
        order      = order[::-1]   # flip so most important is on top

        feat_labels = [self.feature_names[i] for i in order]
        pos_vals    = pos_mean[order]
        neg_vals    = neg_mean[order]

        y_pos = np.arange(len(order))

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(y_pos, pos_vals,  color='#d73027', edgecolor='white', height=0.6, label='Positive contribution')
        ax.barh(y_pos, -neg_vals, color='#4575b4', edgecolor='white', height=0.6, label='Negative contribution')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=10)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Mean SHAP contribution', fontsize=11)
        ax.set_title('Feature Contributions  (red = positive  |  blue = negative)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)

        # Annotate bar values
        for i, (p, n) in enumerate(zip(pos_vals, neg_vals)):
            if p > 0:
                ax.text(p + 0.001, i, f'+{p:.3f}', va='center', fontsize=8, color='#d73027')
            if n > 0:
                ax.text(-n - 0.001, i, f'-{n:.3f}', va='center', ha='right', fontsize=8, color='#4575b4')

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

    def feature_importance_plot(self, sv, class_idx=1, top_n_features=20, show=True, figsize=(10, 7)):
        """
        Pseudo-global feature importance: mean absolute SHAP value per feature
        across all samples — equivalent to LIME's global_importance_plot.

        Mirrors the same style and behavior: returns a DataFrame, respects
        show=False (closes the figure without rendering in Jupyter).

        Parameters
        ----------
        X         : array-like or DataFrame
        class_idx : int   — class to analyse for multiclass models (default=1)
        top_n_features     : int   — max features to display
        show      : bool
        figsize   : tuple

        Returns
        -------
        pd.DataFrame  columns=['feature', 'mean_abs_shap']
        """
        importance = self.top_features(sv, top_n_features)
        feat_labels = importance["feature"].values
        feat_vals = importance["mean_abs_shap"].values

        # Plot
        y_pos = np.arange(len(feat_labels))
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(y_pos, feat_vals, color='#2166ac', edgecolor='white', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=10)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|', fontsize=11)
        ax.set_title('Global Feature Importance (SHAP)', fontsize=13, fontweight='bold')

        # Annotate bars
        for i, v in enumerate(feat_vals):
            ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=8, color='#2166ac')

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)


    # ------------------------------------------------------------------
    # Plots — SINGLE SAMPLE
    # ------------------------------------------------------------------

    def waterfall_plot(self, X, sample_idx=0, class_idx=1, max_display=10, show=True):
        """
        Waterfall plot for a single sample.

        Parameters
        ----------
        sample_idx : int  — which sample to explain
        class_idx  : int  — which class to show for multiclass models (default=1)
        """
        X_arr    = self._to_array(X)
        instance = X_arr[sample_idx].reshape(1, -1)
        sv       = self.shap_values(instance)

        if self._is_multiclass(sv):
            sv = self._slice_class(sv, class_idx)

        shap.plots.waterfall(sv[0], max_display=max_display, show=show)
