import warnings
import numpy as np
import anndata as ad
import pandas as pd
import pytest
from screenpro.phenoscore import runPhenoScore
from screenpro.phenoscore.phenostat import matrixStat, empiricalPValue, empiricalFDR


def _make_test_matrices():
    """Create small matrices for testing matrixStat."""
    rng = np.random.default_rng(42)
    x = rng.integers(10, 50, size=(5, 3)).astype(float)
    y = rng.integers(50, 100, size=(5, 3)).astype(float)
    return x, y


def test_matrixStat_ttest_col():
    x, y = _make_test_matrices()
    p_values = matrixStat(x, y, test='ttest', level='col')
    assert p_values.shape == (x.shape[0],)
    assert np.all((p_values >= 0) & (p_values <= 1) | np.isnan(p_values))


def test_matrixStat_ttest_all():
    x, y = _make_test_matrices()
    p_value = matrixStat(x, y, test='ttest', level='all')
    assert np.isscalar(p_value) or p_value.ndim == 0
    assert (0 <= float(p_value) <= 1) or np.isnan(p_value)


def test_matrixStat_MW_col():
    x, y = _make_test_matrices()
    p_values = matrixStat(x, y, test='MW', level='col')
    assert p_values.shape == (x.shape[0],)
    assert np.all((p_values >= 0) & (p_values <= 1) | np.isnan(p_values))


def test_matrixStat_MW_all():
    x, y = _make_test_matrices()
    p_value = matrixStat(x, y, test='MW', level='all')
    assert np.isscalar(p_value) or p_value.ndim == 0
    assert (0 <= float(p_value) <= 1) or np.isnan(p_value)


def test_matrixStat_KS_col():
    x, y = _make_test_matrices()
    p_values = matrixStat(x, y, test='KS', level='col')
    assert p_values.shape == (x.shape[0],)
    assert np.all((p_values >= 0) & (p_values <= 1) | np.isnan(p_values))


def test_matrixStat_KS_all():
    x, y = _make_test_matrices()
    p_value = matrixStat(x, y, test='KS', level='all')
    assert np.isscalar(p_value) or p_value.ndim == 0
    assert (0 <= float(p_value) <= 1) or np.isnan(p_value)


def test_matrixStat_small_sample_no_warning():
    """Verify SmallSampleWarning is not emitted for small samples."""
    # Use values with varied differences after log-transform to avoid
    # precision-loss warnings; the key check is that SmallSampleWarning
    # (message: 'too small') is not raised
    x = np.array([[10.0, 20.0]])
    y = np.array([[50.0, 300.0]])
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*too small.*', category=RuntimeWarning)
        p_value = matrixStat(x, y, test='ttest', level='all')
    assert p_value is not None
    assert np.isnan(p_value) or (0 <= float(p_value) <= 1)


def test_matrixStat_nan_returns_nan():
    """Verify NaN inputs produce NaN output for MW and KS tests."""
    x = np.array([[np.nan, np.nan]])
    y = np.array([[np.nan, np.nan]])
    assert np.isnan(matrixStat(x, y, test='MW', level='all', transform=None))
    assert np.isnan(matrixStat(x, y, test='KS', level='all', transform=None))


def test_matrixStat_invalid_test():
    x, y = _make_test_matrices()
    with pytest.raises(ValueError, match='not recognized'):
        matrixStat(x, y, test='invalid_test', level='col')


def test_runPhenoScore_MW():
    """Verify runPhenoScore works with Mann-Whitney U test."""
    rng = np.random.default_rng(0)
    cond_A = rng.integers(10, 30, size=(3, 10))
    cond_B = rng.integers(50, 100, size=(3, 10))

    adat = ad.AnnData(
        X=np.concatenate([cond_A, cond_B], axis=0).astype(float),
        obs=pd.DataFrame(
            {'condition': ['A'] * 3 + ['B'] * 3},
            index=pd.Index(['sample_' + str(i) for i in range(6)], name='sample')
        ),
        var=pd.DataFrame(
            {
                'target': ['targetID_' + str(i) for i in range(10)],
                'targetType': ['gene'] * 8 + ['negative_control'] * 2
            },
            index=pd.Index(['targetID_' + str(i) for i in range(10)], name='target')
        )
    )

    result_name, result = runPhenoScore(
        adata=adat,
        cond_ref='A',
        cond_test='B',
        test='MW',
        score_level='compare_reps',
        growth_rate=1,
        n_reps=3,
        ctrl_label='negative_control'
    )

    assert result_name == 'B_vs_A'
    assert isinstance(result, pd.DataFrame)
    assert 'MW pvalue' in result.columns


def test_runPhenoScore():

    # create test data
    cond_A = np.random.randint(0, 30, size=(3, 10))
    cond_B = np.random.randint(20, 100, size=(3, 10))

    adat = ad.AnnData(
        X=np.concatenate([cond_A, cond_B], axis=0),
        obs=pd.DataFrame(
            {
                'condition': ['A'] * 3 + ['B'] * 3
            },
            index=pd.Index(['sample_' + str(i) for i in range(6)], name='sample')
        ),
        var=pd.DataFrame(
            {
                'target': ['targetID_' + str(i) for i in range(10)],
                'sequence': [
                    'ATGCGTACATGTATGCGTG',
                    'ATGCGTATGCATATGCGTC',
                    'ATGCGTATGCGTCATCGTG',
                    'ATGCGTATGCGTATGCATC',
                    'CATCGTATGCGTATGCGTG',
                    'ATGCGTACATGTATGCGTG',
                    'ATGCGTATGCATATGCGTC',
                    'ATGCGTATGCGTCATCGTG',
                    'ATGCGTATGCGTATGCATC',
                    'CATCGTATGCGTATGCGTG'
                ],
                'targetType': ['gene'] * 8 + ['negative_control'] * 2
            },
            index=pd.Index(['targetID_' + str(i) for i in range(10)], name='target')
        )
    )

    print(adat.to_df())

    assert isinstance(adat, ad.AnnData)

    # run function
    result_name, result = runPhenoScore(
        adata=adat,
        cond_ref='A',
        cond_test='B',
        test='ttest',
        score_level='compare_reps',
        growth_rate=1,
        n_reps=2,
        ctrl_label='negative_control'
    )

    # check result name
    assert result_name == 'B_vs_A'

    # check result dataframe
    assert isinstance(result, pd.DataFrame)

# ── empiricalPValue tests ──────────────────────────────────────────────────────

def _make_score_arrays(seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(20)
    null_scores = rng.standard_normal(100)
    return scores, null_scores


def test_empiricalPValue_two_sided_range():
    scores, null_scores = _make_score_arrays()
    p = empiricalPValue(scores, null_scores, tail='two-sided')
    assert p.shape == (len(scores),)
    assert np.all((p > 0) & (p <= 1))


def test_empiricalPValue_less_range():
    scores, null_scores = _make_score_arrays()
    p = empiricalPValue(scores, null_scores, tail='less')
    assert np.all((p > 0) & (p <= 1))


def test_empiricalPValue_greater_range():
    scores, null_scores = _make_score_arrays()
    p = empiricalPValue(scores, null_scores, tail='greater')
    assert np.all((p > 0) & (p <= 1))


def test_empiricalPValue_nan_input():
    scores = np.array([1.0, np.nan, -1.0])
    null_scores = np.array([0.5, -0.5, 0.0])
    p = empiricalPValue(scores, null_scores)
    assert np.isnan(p[1])
    assert not np.isnan(p[0])
    assert not np.isnan(p[2])


def test_empiricalPValue_empty_null():
    scores = np.array([1.0, 2.0])
    null_scores = np.array([])
    p = empiricalPValue(scores, null_scores)
    assert np.all(np.isnan(p))


def test_empiricalPValue_invalid_tail():
    scores, null_scores = _make_score_arrays()
    with pytest.raises(ValueError, match='not recognized'):
        empiricalPValue(scores, null_scores, tail='invalid')


def test_empiricalPValue_extreme_score_low_pvalue():
    """A score far from the null distribution should have a small p-value."""
    null_scores = np.zeros(1000)
    extreme_score = np.array([100.0])
    p = empiricalPValue(extreme_score, null_scores, tail='two-sided')
    assert p[0] < 0.01


# ── empiricalFDR tests ─────────────────────────────────────────────────────────

def test_empiricalFDR_range():
    scores, null_scores = _make_score_arrays()
    fdr = empiricalFDR(scores, null_scores, tail='two-sided')
    assert fdr.shape == (len(scores),)
    assert np.all(((fdr >= 0) & (fdr <= 1)) | np.isnan(fdr))


def test_empiricalFDR_nan_input():
    scores = np.array([1.0, np.nan, -1.0])
    null_scores = np.array([0.5, -0.5, 0.0])
    fdr = empiricalFDR(scores, null_scores)
    assert np.isnan(fdr[1])
    assert not np.isnan(fdr[0])
    assert not np.isnan(fdr[2])


def test_empiricalFDR_empty_null():
    scores = np.array([1.0, 2.0])
    null_scores = np.array([])
    fdr = empiricalFDR(scores, null_scores)
    assert np.all(np.isnan(fdr))


def test_empiricalFDR_invalid_tail():
    scores, null_scores = _make_score_arrays()
    with pytest.raises(ValueError, match='not recognized'):
        empiricalFDR(scores, null_scores, tail='invalid')


def test_empiricalFDR_extreme_score_low_fdr():
    """Scores far from the null distribution should yield low FDR."""
    rng = np.random.default_rng(42)
    null_scores = rng.standard_normal(1000)
    scores = np.array([10.0, 11.0, 12.0])
    fdr = empiricalFDR(scores, null_scores, tail='two-sided')
    assert np.all(fdr < 0.05)


# ── runPhenoScore new columns ──────────────────────────────────────────────────

def test_runPhenoScore_has_empirical_columns():
    """Verify runPhenoScore result includes empirical pvalue and FDR columns."""
    rng = np.random.default_rng(7)
    cond_A = rng.integers(10, 30, size=(3, 10))
    cond_B = rng.integers(50, 100, size=(3, 10))

    adat = ad.AnnData(
        X=np.concatenate([cond_A, cond_B], axis=0).astype(float),
        obs=pd.DataFrame(
            {'condition': ['A'] * 3 + ['B'] * 3},
            index=pd.Index(['sample_' + str(i) for i in range(6)], name='sample')
        ),
        var=pd.DataFrame(
            {
                'target': ['targetID_' + str(i) for i in range(10)],
                'targetType': ['gene'] * 8 + ['negative_control'] * 2
            },
            index=pd.Index(['targetID_' + str(i) for i in range(10)], name='target')
        )
    )

    _, result = runPhenoScore(
        adata=adat,
        cond_ref='A',
        cond_test='B',
        test='ttest',
        score_level='compare_reps',
        growth_rate=1,
        n_reps=3,
        ctrl_label='negative_control'
    )

    assert 'empirical pvalue' in result.columns
    assert 'empirical FDR' in result.columns
    emp_p = result['empirical pvalue'].dropna()
    emp_fdr = result['empirical FDR'].dropna()
    assert np.all((emp_p > 0) & (emp_p <= 1))
    assert np.all((emp_fdr >= 0) & (emp_fdr <= 1))
