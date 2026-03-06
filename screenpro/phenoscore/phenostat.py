"""
phenostat module: internal module for statistical analysis of phenoscore data.
"""

from scipy.stats import ttest_rel, mannwhitneyu, ks_2samp
import numpy as np
from statsmodels.stats.multitest import multipletests
import warnings


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
    if test == 'MW':
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

    elif test == 'KS':
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

    elif test == 'ttest':
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


def empiricalFDR():
    pass
