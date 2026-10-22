import warnings
import numpy as np
import anndata as ad
import pandas as pd
import pytest
from screenpro.phenoscore import runPhenoScore
from screenpro.phenoscore.phenostat import matrixStat, controlStat, multipleTestsCorrection


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

# ── runPhenoScore new columns ──────────────────────────────────────────────────

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


def test_drawVolcano_keeps_controls_and_colors_hits_despite_nan_in_other_columns():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from screenpro.assays import PooledScreens
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        'target': [f'g{i}' for i in range(n - 100)] + ['negative_control'] * 100,
        'score': rng.normal(0, 0.2, n),
        'pvalue': rng.uniform(0.01, 1, n),
        'other': np.where(np.arange(n) >= n - 100, np.nan, 1.0),   # NaN on every control row
    })
    df.loc[0, ['score', 'pvalue']] = [-3.0, 0.0]                  # strong hit with p == 0
    df.loc[1, 'score'] = np.nan                                    # untestable row
    screen = PooledScreens(ad.AnnData(X=np.ones((2, 2)), obs=pd.DataFrame(index=['a', 'b'])), test='modT')
    screen.phenotypes['r'] = {'config': {}, 'results': {'rho:B_vs_A': df}}
    fig, ax = plt.subplots()
    screen.drawVolcano(ax, 'rho:B_vs_A', threshold=5, run_name='r')
    counts = {h.get_label(): len(h.get_offsets()) for h in ax.collections}
    assert counts['negative_control'] == 100
    assert counts['sensitivity_hit'] >= 1
    assert all(np.isfinite(ax.get_ylim()))
    plt.close(fig)


def test_drawVolcano_auto_pvalue_col_follows_run_test():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from screenpro.assays import PooledScreens
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'target': [f'g{i}' for i in range(50)] + ['negative_control'] * 50,
        'score': rng.normal(0, 0.2, 100),
        'modT pvalue': rng.uniform(0.01, 1, 100),
    })
    screen = PooledScreens(ad.AnnData(X=np.ones((2, 2)), obs=pd.DataFrame(index=['a', 'b'])), test='modT')
    screen.phenotypes['r'] = {'config': {'test': 'modT'}, 'results': {'rho:B_vs_A': df}}
    fig, ax = plt.subplots()
    screen.drawVolcano(ax, 'rho:B_vs_A', threshold=5, run_name='r')       # no pvalue_col given
    assert 'log' in ax.get_ylabel() and '10' in ax.get_ylabel()
    assert 'moderated t-test' in [t.get_text() for t in ax.texts]          # full name, not 'modT'
    plt.close(fig)

    screen.phenotypes['r']['config'] = {'test': 'MW'}                       # column for MW is absent
    with pytest.raises(ValueError, match='No p-value column found'):
        screen.drawVolcano(plt.subplots()[1], 'rho:B_vs_A', threshold=5, run_name='r')

    df2 = df.rename(columns={'modT pvalue': 'pvalue'})                      # e.g. DESeq2 run, no test recorded
    screen.phenotypes['r'] = {'config': {}, 'results': {'rho:B_vs_A': df2}}
    screen.drawVolcano(plt.subplots()[1], 'rho:B_vs_A', threshold=5, run_name='r')
