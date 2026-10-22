"""
phenostat module: internal module for statistical analysis of phenoscore data.
"""

from scipy.stats import ttest_rel, mannwhitneyu, ks_2samp, t as t_dist
from scipy.special import digamma, polygamma
import numpy as np
from statsmodels.stats.multitest import multipletests
import warnings
from math import comb


## Registry of statistical tests exposed to users

# canonical name -> (aliases, one-line description shown in error messages / docs)
TESTS = {
    'modT':  (('moderated-t', 'limma'),
              "reproducible across biological replicates (moderated t-test on replicate scores)"),
    'MW':    (('Mann-Whitney',),
              "gene's guides differ from non-targeting guides (Mann-Whitney U rank test)"),
    'KS':    (('Kolmogorov-Smirnov',),
              "guide score distribution differs from non-targeting guides (Kolmogorov-Smirnov test)"),
    'ttest': (('TT', 'T-test'),
              "legacy paired t-test on log10 counts (pseudoreplicated; use to reproduce v0.7 results)"),
}


# canonical name -> full name for plot labels and reports
TEST_NAMES = {
    'modT': 'moderated t-test',
    'MW': 'Mann\u2013Whitney U test',
    'KS': 'Kolmogorov\u2013Smirnov test',
    'ttest': 'paired t-test (legacy)',
}


def validateTest(test):
    """Check a user-supplied ``test`` name and return its canonical form.

    ``test`` has no default anywhere in the public API: the statistical test
    must match the question being asked, so the user has to choose one.

    Parameters:
        test (str): test name or alias (see ``TESTS``)

    Returns:
        str: canonical test name

    Raises:
        ValueError: if ``test`` is None or not recognized, with the list of options
    """
    if test is not None:
        for name, (aliases, _) in TESTS.items():
            if test == name or test in aliases:
                return name

    options = '\n'.join(f"  {name!r:8} {desc}" for name, (_, desc) in TESTS.items())
    problem = '`test` is required.' if test is None else f'Test {test!r} not recognized.'
    raise ValueError(
        f"{problem} Choose the test that matches your question:\n{options}\n"
        "BH-adjusted p-values are reported alongside the raw p-values."
    )


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

    - ``MW``: Mann-Whitney U rank test (non-parametric, *independent* samples).
      Does not assume normality. Caveats for this function: (1) ``x`` and ``y`` here
      are paired (same guide, same replicate), but MW ignores that pairing, so
      guide-to-guide baseline abundance dominates the variance and power is poor;
      (2) with n replicates per condition the exact two-sided p-value cannot go
      below ``2 / C(2n, n)`` (0.1 for n=3), so no element can pass a 0.05 cutoff
      (a warning is emitted). For gene-level MW use :func:`controlStat`, which
      compares guide-level scores against non-targeting guide scores instead.

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
        if level in ('col', 'row'):
            _warn_if_exact_floor(n_x=x.shape[1 if level == 'col' else 0],
                                 n_y=y.shape[1 if level == 'col' else 0])
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


## Empirical-Bayes moderated t-test (limma-style, Smyth 2004)

def _trigammaInverse(x, tol=1e-8, max_iter=50):
    """Solve ``trigamma(y) = x`` for y by Newton iteration (as in limma::trigammaInverse)."""
    if np.isnan(x):
        return np.nan
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(max_iter):
        tri = polygamma(1, y)
        dif = tri * (1 - tri / x) / polygamma(2, y)
        y = y + dif
        if -dif / y < tol:
            break
    return y


def fitPriorVariance(s2, df):
    """Fit a scaled inverse chi-square prior to per-element sample variances.

    Model (Smyth 2004): element variances ``sigma_g^2 ~ s0^2 * d0 / chi2_{d0}`` and
    sample variances ``s_g^2 | sigma_g^2 ~ sigma_g^2 * chi2_{d_g} / d_g``. Marginally
    ``s_g^2 / s0^2 ~ F(d_g, d0)``. Parameters are estimated by the method of moments
    on ``z = log(s2) - digamma(d/2) + log(d/2)``, which has mean
    ``log(s0^2) - digamma(d0/2) + log(d0/2)`` and variance
    ``trigamma(d/2) + trigamma(d0/2)``.

    Parameters:
        s2 (np.array): sample variances
        df (np.array): residual degrees of freedom of each variance

    Returns:
        tuple: (s0_sq, d0); ``d0`` is ``np.inf`` when variances are homogeneous
    """
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    ok = np.isfinite(s2) & (df > 0)
    s2, df = s2[ok], df[ok]
    if len(s2) < 2:
        return np.nan, np.nan

    # guard against exact zeros (e.g. identical replicates) before taking logs
    s2 = np.maximum(s2, 1e-5 * np.median(s2)) if np.median(s2) > 0 else np.maximum(s2, 1e-10)

    z = np.log(s2) - digamma(df / 2) + np.log(df / 2)
    z_mean = z.mean()
    z_var = np.sum((z - z_mean) ** 2) / (len(z) - 1) - np.mean(polygamma(1, df / 2))

    if z_var > 0:
        d0 = 2 * _trigammaInverse(z_var)
        s0_sq = np.exp(z_mean + digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0 = np.inf
        s0_sq = np.exp(z_mean)
    return s0_sq, d0


def moderatedTTest(scores, return_details=False):
    """Moderated one-sample t-test of per-replicate scores against zero.

    Each row of ``scores`` is one element (guide, transcript or gene) and each column
    one biological replicate. Scores are expected to be on the phenotype scale
    (e.g. delta/gamma/tau/rho) and centred on non-targeting controls, so that the
    null hypothesis is a mean of 0.

    With only 2-3 replicates the per-element sample variance is very noisy: elements
    that happen to have tightly agreeing replicates get huge t-statistics. The
    empirical-Bayes step shrinks each variance towards a prior fitted across all
    elements (see :func:`fitPriorVariance`):

    .. math::

        \\tilde{s}_g^2 = \\frac{d_0 s_0^2 + d_g s_g^2}{d_0 + d_g}, \\qquad
        t_g = \\frac{\\bar{x}_g}{\\tilde{s}_g / \\sqrt{n_g}} \\sim t_{d_0 + d_g}

    so the test gains ``d0`` extra degrees of freedom. Elements with fewer than 2
    finite replicates get NaN.

    Parameters:
        scores (np.array): matrix of shape (n_elements, n_replicates); NaN/inf are ignored
        return_details (bool): also return dict with t, df, moderated variance, prior

    Returns:
        np.array: two-sided p-values (or ``(p_values, details)``)
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError('scores must be a 2D (elements x replicates) array')
    scores = np.where(np.isfinite(scores), scores, np.nan)

    n = np.sum(~np.isnan(scores), axis=1)
    df = n - 1
    usable = n >= 2

    mean = np.full(len(scores), np.nan)
    s2 = np.full(len(scores), np.nan)
    mean[usable] = np.nanmean(scores[usable], axis=1)
    s2[usable] = np.nanvar(scores[usable], axis=1, ddof=1)

    s0_sq, d0 = fitPriorVariance(s2[usable], df[usable])

    if np.isnan(d0):  # too few elements to fit a prior: fall back to ordinary t
        d0, s0_sq = 0.0, 0.0
    if np.isinf(d0):
        s2_post = np.full(len(scores), s0_sq)
        df_total = np.full(len(scores), np.inf)
    else:
        s2_post = (d0 * s0_sq + df * s2) / (d0 + df)
        df_total = d0 + df

    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = mean / np.sqrt(s2_post / n)
    p_values = 2 * t_dist.sf(np.abs(t_stat), df_total)
    p_values[~usable] = np.nan

    if return_details:
        return p_values, {
            't': t_stat, 'df': df_total, 'mean': mean, 'var_moderated': s2_post,
            'var_prior': s0_sq, 'df_prior': d0,
        }
    return p_values


def multipleTestsCorrection(p_values, method='fdr_bh'):
    """
    Calculate adjusted p-values using multiple testing correction.

    Tests that could not be run (NaN p-values) are excluded from the correction,
    so they do not inflate the number of hypotheses, and get NaN adjusted p-values.
    The input array is not modified.

    Parameters:
        p_values (np.array): array of p-values
        method (str): method to use for multiple testing correction
    
    Returns:
        np.array: array of adjusted p-values
    """
    if p_values is None:
        raise ValueError('p_values is None')
    if method != 'fdr_bh':
        raise ValueError(f'Method "{method}" not recognized')

    p_values = np.asarray(p_values, dtype=float)
    adj_p_values = np.full(p_values.shape, np.nan)
    valid = ~np.isnan(p_values)
    if valid.any():
        adj_p_values[valid] = multipletests(p_values[valid], alpha=0.05, method='fdr_bh')[1]

    return adj_p_values


def controlStat(scores, null_scores, test='MW'):
    """Test guide-level scores of one target against non-targeting guide scores.

    Unlike :func:`matrixStat`, the compared groups are genuinely independent
    (guides of a target vs. non-targeting guides) and the unit of replication is
    the guide, with replicates already collapsed into one score per guide. This
    avoids both the unpaired-on-paired-data problem and pseudoreplication
    (guides x replicates entries treated as independent observations).

    Parameters:
        scores (array-like): per-guide scores of the target, shape (n_guides,) or
            (n_guides, n_reps); replicates are averaged per guide
        null_scores (array-like): per-guide scores of non-targeting controls, same layout
        test (str): 'MW' (Mann-Whitney U) or 'KS' (Kolmogorov-Smirnov)

    Returns:
        float: p-value (NaN if either group has fewer than 2 guides)
    """
    def _per_guide(a):
        a = np.asarray(a, dtype=float)
        return np.nanmean(a, axis=1) if a.ndim == 2 else a

    if test in ('MW', 'Mann-Whitney'):
        return _mannwhitneyu_safe(_per_guide(scores), _per_guide(null_scores))
    elif test in ('KS', 'Kolmogorov-Smirnov'):
        return _ks_2samp_safe(_per_guide(scores), _per_guide(null_scores))
    raise ValueError(f'Test "{test}" not supported for control comparison')


def empiricalPValue(scores, null_scores, tail='two-sided'):
    """Empirical p-values against a null distribution of non-targeting control scores.

    TODO: design pending. Open points before implementing: size-matched null for targets
    with different guide counts, overlapping pseudogenes, seeding, and how a two-sided
    value should treat an asymmetric null.
    """
    pass


def empiricalFDR(scores, null_scores, tail='two-sided'):
    """Empirical FDR (q-value) against a null distribution of non-targeting control scores.

    TODO: design pending; see :func:`empiricalPValue`.
    """
    pass



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


def _warn_if_exact_floor(n_x, n_y, alpha=0.05):
    """Warn when the smallest attainable two-sided exact MW p-value exceeds alpha."""
    if n_x < 1 or n_y < 1:
        return
    min_p = 2 / comb(n_x + n_y, n_x)
    if min_p > alpha:
        warnings.warn(
            f'Mann-Whitney with {n_x} vs {n_y} replicates cannot give a two-sided '
            f'p-value below {min_p:.3g}; no element can reach p < {alpha}.',
            UserWarning, stacklevel=3
        )
