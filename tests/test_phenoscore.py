import warnings
import numpy as np
import anndata as ad
import pandas as pd
import pytest
from screenpro.phenoscore import runPhenoScore
from screenpro.phenoscore.phenostat import matrixStat


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