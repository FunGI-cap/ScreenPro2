import warnings
import numpy as np
import anndata as ad
import pandas as pd
import pytest
from screenpro.phenoscore import runPhenoScore
from screenpro.phenoscore.phenostat import matrixStat, controlStat, multipleTestsCorrection, empiricalPValue, empiricalFDR


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


def test_matrixStat_MW_exact_floor_warns():
    """3 vs 3 replicates: smallest two-sided exact MW p-value is 0.1."""
    x = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[10.0, 20.0, 30.0]])
    with pytest.warns(UserWarning, match='cannot give a two-sided'):
        p = matrixStat(x, y, test='MW', level='col', transform=None)
    assert p[0] == pytest.approx(0.1)


def test_controlStat_detects_shifted_guides():
    rng = np.random.default_rng(1)
    null = rng.normal(0, 0.3, size=(500, 3))
    hit = rng.normal(-1.5, 0.3, size=(6, 3))
    assert controlStat(hit, null, test='MW') < 1e-3
    assert controlStat(null[:6], null[6:], test='MW') > 0.01
    assert np.isnan(controlStat(hit[:1], null, test='MW'))


def test_multipleTestsCorrection_nan_and_no_mutation():
    p = np.array([0.001, np.nan, 0.5, 0.02])
    p_orig = p.copy()
    adj = multipleTestsCorrection(p)
    np.testing.assert_array_equal(p, p_orig)  # input untouched
    assert np.isnan(adj[1])
    # NaN excluded from m: same as correcting the 3 valid p-values alone
    np.testing.assert_allclose(adj[[0, 2, 3]], multipleTestsCorrection(np.array([0.001, 0.5, 0.02])))


def test_scoreTargetGroup_top_n_used_for_score_and_pvalue():
    from screenpro.phenoscore.delta import scoreTargetGroup
    rng = np.random.default_rng(3)
    n_g = 6
    ref = pd.DataFrame(rng.integers(100, 200, (n_g + 50, 3)).astype(float))
    test = ref.copy()
    test.iloc[:2] *= 8            # 2 strong guides in the target
    grp = pd.DataFrame(index=ref.index[:n_g])
    ctrl = ref.iloc[n_g:].to_numpy(); ctrl_t = test.iloc[n_g:].to_numpy()
    s_all, p_all, n_all = scoreTargetGroup(grp, ref, test, ctrl, ctrl_t, test='MW')
    s_top, p_top, n_top = scoreTargetGroup(grp, ref, test, ctrl, ctrl_t, test='MW', keep_top_n=2)
    assert n_all == n_g and n_top == 2
    assert np.all(np.abs(s_top) > np.abs(s_all))
    assert p_top != p_all


def test_moderatedTTest_recovers_prior_and_is_calibrated():
    from screenpro.phenoscore.phenostat import moderatedTTest
    rng = np.random.default_rng(0)
    G, d0, s0sq = 4000, 6.0, 0.15 ** 2
    sig2 = s0sq * d0 / rng.chisquare(d0, G)
    x = rng.normal(size=(G, 3)) * np.sqrt(sig2)[:, None]       # all null
    p, det = moderatedTTest(x, return_details=True)
    assert 4 < det['df_prior'] < 10
    assert det['var_prior'] == pytest.approx(s0sq, rel=0.2)
    assert np.mean(p < 0.05) == pytest.approx(0.05, abs=0.01)


def test_moderatedTTest_more_power_than_ordinary_t():
    from scipy import stats
    from screenpro.phenoscore.phenostat import moderatedTTest
    rng = np.random.default_rng(1)
    G = 3000
    sig2 = 0.15 ** 2 * 6 / rng.chisquare(6, G)
    mu = np.zeros(G); mu[:150] = 0.35
    x = mu[:, None] + rng.normal(size=(G, 3)) * np.sqrt(sig2)[:, None]
    p_mod = moderatedTTest(x)
    p_ord = stats.ttest_1samp(x, 0, axis=1).pvalue
    assert np.sum(p_mod[:150] < 1e-3) > 5 * max(1, np.sum(p_ord[:150] < 1e-3))


def test_moderatedTTest_nan_handling():
    from screenpro.phenoscore.phenostat import moderatedTTest
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 3))
    x[0, 1:] = np.nan          # only 1 replicate -> NaN
    x[1, 0] = np.inf           # non-finite ignored, 2 replicates remain
    p = moderatedTTest(x)
    assert np.isnan(p[0]) and np.isfinite(p[1])


@pytest.mark.parametrize('level', ['compare_reps', 'compare_guides'])
def test_runPhenoScore_modT(level):
    rng = np.random.default_rng(5)
    n_g, n_c, per = 30, 60, 4
    n_el = n_g * per + n_c
    base = rng.integers(200, 2000, n_el).astype(float)
    eff = np.ones(n_el); eff[:per * 3] = 0.5                    # first 3 genes depleted
    A = np.vstack([base * rng.lognormal(0, .1, n_el) for _ in range(3)])
    B = np.vstack([base * eff * rng.lognormal(0, .1, n_el) for _ in range(3)])
    names = [f'g{i // per}_{i % per}' for i in range(n_g * per)] + [f'nc_{i}' for i in range(n_c)]
    adat = ad.AnnData(
        X=np.concatenate([A, B]),
        obs=pd.DataFrame({'condition': ['A'] * 3 + ['B'] * 3, 'replicate': [1, 2, 3] * 2},
                         index=[f's{i}' for i in range(6)]),
        var=pd.DataFrame({
            'target': [f'g{i // per}' for i in range(n_g * per)] + ['negative_control'] * n_c,
            'targetType': ['gene'] * (n_g * per) + ['negative_control'] * n_c}, index=names))
    _, res = runPhenoScore(adat, cond_ref='A', cond_test='B', test='modT', score_level=level,
                           growth_rate=1, n_reps=3, ctrl_label='negative_control',
                           num_pseudogenes=20, pseudogene_size=4)
    col = 'modT pvalue'
    assert col in res.columns and res[col].notna().any()
    hits = res[res['score'] < -0.5]
    assert len(hits) >= 3 and (hits[col] < 0.01).all()


def test_empiricalPValue_two_sided_matches_one_sided_on_skewed_null():
    rng = np.random.default_rng(0)
    null = rng.gamma(2, 0.15, 2000) - 0.3
    scores = np.array([-0.25, 0.0, 0.8])
    p2 = empiricalPValue(scores, null, 'two-sided')
    lo = empiricalPValue(scores, null, 'less'); hi = empiricalPValue(scores, null, 'greater')
    np.testing.assert_allclose(p2, np.minimum(2 * np.minimum(lo, hi), 1))


def test_empiricalPValue_calibrated_under_null():
    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, 5000)
    p = empiricalPValue(rng.normal(0, 1, 20000), null)
    assert np.mean(p < 0.05) == pytest.approx(0.05, abs=0.008)


def test_empiricalFDR_excludes_null_from_tested_set_and_is_monotone():
    rng = np.random.default_rng(2)
    null = rng.normal(0, 1, 2000)
    genes = np.r_[rng.normal(0, 1, 800), rng.normal(-4, 0.5, 100)]
    scores = np.r_[genes, rng.normal(0, 1, 500)]
    is_null = np.r_[np.zeros(len(genes), bool), np.ones(500, bool)]
    fdr = empiricalFDR(scores, null, is_null=is_null)
    assert np.isnan(fdr[is_null]).all() and not np.isnan(fdr[~is_null]).any()
    g = fdr[~is_null]; s = genes
    assert np.all(np.diff(g[s < 0][np.argsort(np.abs(s[s < 0]))]) <= 1e-12)   # more extreme -> lower q
    assert np.median(g[800:]) < 0.01          # the 100 strong hits are called
    assert np.mean(g[:800] < 0.05) < 0.1      # nulls rarely are


def test_validateTest_required_and_canonical():
    from screenpro.phenoscore.phenostat import validateTest, TESTS
    with pytest.raises(ValueError, match=r"(?s)`test` is required.*modT.*MW.*ttest") as e:
        validateTest(None)
    for name, (_, desc) in TESTS.items():
        assert desc in str(e.value)
    with pytest.raises(ValueError, match="not recognized"):
        validateTest('wilcoxon')
    assert validateTest('limma') == 'modT' and validateTest('TT') == 'ttest' and validateTest('MW') == 'MW'


def test_runPhenoScore_and_PooledScreens_require_test():
    from screenpro.assays import PooledScreens
    rng = np.random.default_rng(0)
    adat = ad.AnnData(X=rng.integers(10, 100, (6, 10)).astype(float),
                      obs=pd.DataFrame({'condition': ['A'] * 3 + ['B'] * 3}, index=[f's{i}' for i in range(6)]),
                      var=pd.DataFrame({'target': [f't{i}' for i in range(10)],
                                        'targetType': ['gene'] * 8 + ['negative_control'] * 2},
                                       index=[f't{i}' for i in range(10)]))
    with pytest.raises(ValueError, match="`test` is required"):
        runPhenoScore(adat, cond_ref='A', cond_test='B', score_level='compare_reps')
    with pytest.raises(ValueError, match="`test` is required"):
        PooledScreens(adat)
    assert PooledScreens(adat, test='limma').test == 'modT'
