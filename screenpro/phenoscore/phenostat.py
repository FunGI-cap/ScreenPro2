"""
phenostat module: internal module for statistical analysis of phenoscore data.
"""

from scipy.stats import ttest_rel, mannwhitneyu, ks_2samp
import numpy as np
from statsmodels.stats.multitest import multipletests
import warnings


## Core functions for statistical analysis of phenoscore data

def matrixStat(x, y, test, level, transform='log10'):
    """
    Get p-values comparing `y` vs `x` matrices.

    The choice of statistical test should reflect the data characteristics:

    - ``ttest``: Paired t-test (parametric). Assumes that the differences between
      paired observations are normally distributed. Well-suited when replicates are
      paired (same guide/construct measured across conditions) and sample size is
      sufficient (n >= 3 pairs). This is the standard choice for CRISPR screens
      with biological replicates.

    - ``MW``: Mann-Whitney U rank test (non-parametric). Does not assume normality
      and is more robust for small samples or skewed distributions. Appropriate when
      comparing groups of guide elements targeting the same gene, where values may
      not be normally distributed or sample sizes are too small for the t-test.

    - ``KS``: Kolmogorov-Smirnov test (non-parametric). Tests whether two samples
      are drawn from the same continuous distribution, sensitive to differences in
      shape, location, and scale. Useful for detecting distributional differences
      beyond the mean, e.g., when comparing guide-level distributions across
      conditions.

    Parameters:
        x (np.array): array of values
        y (np.array): array of values
        test (str): test to use for calculating p-value ('ttest', 'MW', or 'KS')
        level (str): level at which to calculate p-value ('col', 'row', or 'all')
        transform (str): transformation to apply to values before running test
    
    Returns:
        np.array: array of p-values
    """
    # log-transform values
    if transform is None:
        pass
    elif transform == 'log10':
        x = np.log10(x)
        y = np.log10(y)
    else:
        raise ValueError(f'Transform "{transform}" not recognized')
    
    # calculate p-values
    if test == 'MW' or test == 'Mann-Whitney':
        # run Mann-Whitney U rank test (non-parametric, independent samples)
        if level == 'col':
            p_value = np.array([
                _mannwhitneyu_safe(y[i, :], x[i, :])
                for i in range(y.shape[0])
            ])
        elif level == 'row':
            p_value = np.array([
                _mannwhitneyu_safe(y[:, j], x[:, j])
                for j in range(y.shape[1])
            ])
        elif level == 'all':
            p_value = _mannwhitneyu_safe(y.flatten(), x.flatten())
        else:
            raise ValueError(f'Level "{level}" not recognized')
        return p_value

    elif test == 'KS' or test == 'Kolmogorov-Smirnov':
        # run Kolmogorov-Smirnov test (non-parametric, two-sample)
        if level == 'col':
            p_value = np.array([
                _ks_2samp_safe(y[i, :], x[i, :])
                for i in range(y.shape[0])
            ])
        elif level == 'row':
            p_value = np.array([
                _ks_2samp_safe(y[:, j], x[:, j])
                for j in range(y.shape[1])
            ])
        elif level == 'all':
            p_value = _ks_2samp_safe(y.flatten(), x.flatten())
        else:
            raise ValueError(f'Level "{level}" not recognized')
        return p_value

    elif test == 'TT' or test == 'ttest' or test == 'T-test':
        # run paired t-test (parametric); suppress SmallSampleWarning since small
        # sample sizes (e.g. 2-3 replicates) are expected in CRISPR screens
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*too small.*', category=RuntimeWarning)
            if level == 'col':
                p_value = ttest_rel(y, x, axis=1, nan_policy='omit')[1]
            elif level == 'row':
                p_value = ttest_rel(y, x, axis=0, nan_policy='omit')[1]
            elif level == 'all':
                # flatten across all values
                p_value = ttest_rel(y, x, axis=None, nan_policy='omit')[1]
            else:
                raise ValueError(f'Level "{level}" not recognized')
        return p_value
    else:
        raise ValueError(f'Test "{test}" not recognized')


def multipleTestsCorrection(p_values, method='fdr_bh'):
    """
    Calculate adjusted p-values using multiple testing correction.

    Parameters:
        p_values (np.array): array of p-values
        method (str): method to use for multiple testing correction
    
    Returns:
        np.array: array of adjusted p-values
    """
    if method == 'fdr_bh':
        # fill na with 1
        p_values[np.isnan(p_values)] = 1
        # Calculate the adjusted p-values using the Benjamini-Hochberg method
        if p_values is None:
            raise ValueError('p_values is None')
        _, adj_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    else:
        raise ValueError(f'Method "{method}" not recognized')

    return adj_p_values


def empiricalPValue(scores, null_scores, tail='two-sided'):
    """Calculate empirical p-values by comparing scores to a null distribution.

    For each score in ``scores``, the empirical p-value is the proportion of
    null scores that are at least as extreme as the observed score. A pseudocount
    of 1 is added to both numerator and denominator to avoid p-values of zero
    and to account for the finite size of the null distribution.

    Parameters:
        scores (array-like): observed phenotype scores
        null_scores (array-like): null/non-targeting control scores
        tail (str): 'two-sided' (default), 'less', or 'greater'

    Returns:
        np.array: array of empirical p-values in [0, 1]
    """
    scores = np.asarray(scores, dtype=float)
    null_scores = np.asarray(null_scores, dtype=float)
    null_scores = null_scores[~np.isnan(null_scores)]
    n_null = len(null_scores)

    if n_null == 0:
        return np.full(len(scores), np.nan)

    p_values = np.empty(len(scores))

    for i, score in enumerate(scores):
        if np.isnan(score):
            p_values[i] = np.nan
            continue

        if tail == 'two-sided':
            n_extreme = np.sum(np.abs(null_scores) >= np.abs(score))
        elif tail == 'less':
            n_extreme = np.sum(null_scores <= score)
        elif tail == 'greater':
            n_extreme = np.sum(null_scores >= score)
        else:
            raise ValueError(f'Tail "{tail}" not recognized. Use "two-sided", "less", or "greater".')

        # pseudocount of 1 in numerator and denominator
        p_values[i] = (n_extreme + 1) / (n_null + 1)

    return p_values


def empiricalFDR(scores, null_scores, tail='two-sided'):
    """Calculate empirical FDR by comparing scores to a null distribution.

    For each threshold defined by an observed score, the empirical FDR is:

    .. math::

        \\text{FDR}(s) = \\frac{\\text{expected false positives}}{\\text{observed positives}}

    where *expected false positives* = ``(# null extreme) / n_null * n_genes``
    and *observed positives* = ``# gene scores at least as extreme as s``.

    This approach provides direct FDR control using the empirical null distribution
    from non-targeting controls, without relying on parametric assumptions or
    multiple-testing correction procedures.

    Parameters:
        scores (array-like): observed phenotype scores (e.g. targeting genes)
        null_scores (array-like): null/non-targeting control scores
        tail (str): 'two-sided' (default), 'less', or 'greater'

    Returns:
        np.array: array of empirical FDR values clipped to [0, 1]
    """
    scores = np.asarray(scores, dtype=float)
    null_scores = np.asarray(null_scores, dtype=float)
    null_scores = null_scores[~np.isnan(null_scores)]
    n_null = len(null_scores)
    n_genes = np.sum(~np.isnan(scores))

    if n_null == 0 or n_genes == 0:
        return np.full(len(scores), np.nan)

    fdr_values = np.empty(len(scores))

    for i, score in enumerate(scores):
        if np.isnan(score):
            fdr_values[i] = np.nan
            continue

        if tail == 'two-sided':
            n_extreme_null = np.sum(np.abs(null_scores) >= np.abs(score))
            n_extreme_genes = np.sum(np.abs(scores[~np.isnan(scores)]) >= np.abs(score))
        elif tail == 'less':
            n_extreme_null = np.sum(null_scores <= score)
            n_extreme_genes = np.sum(scores[~np.isnan(scores)] <= score)
        elif tail == 'greater':
            n_extreme_null = np.sum(null_scores >= score)
            n_extreme_genes = np.sum(scores[~np.isnan(scores)] >= score)
        else:
            raise ValueError(f'Tail "{tail}" not recognized. Use "two-sided", "less", or "greater".')

        # expected false positives normalized to gene sample size
        expected_fp = (n_extreme_null / n_null) * n_genes

        if n_extreme_genes > 0:
            fdr_values[i] = min(expected_fp / n_extreme_genes, 1.0)
        else:
            fdr_values[i] = 0.0

    return fdr_values



## Utility functions for safe statistical testing with small sample sizes

def _mannwhitneyu_safe(y, x):
    """Run Mann-Whitney U test, returning NaN when sample size is insufficient."""
    y_clean = y[~np.isnan(y)]
    x_clean = x[~np.isnan(x)]
    if len(y_clean) < 2 or len(x_clean) < 2:
        return np.nan
    return mannwhitneyu(y_clean, x_clean, alternative='two-sided')[1]


def _ks_2samp_safe(y, x):
    """Run Kolmogorov-Smirnov test, returning NaN when sample size is insufficient."""
    y_clean = y[~np.isnan(y)]
    x_clean = x[~np.isnan(x)]
    if len(y_clean) < 2 or len(x_clean) < 2:
        return np.nan
    return ks_2samp(y_clean, x_clean)[1]
