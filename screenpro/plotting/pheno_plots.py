import numpy as np
import pandas as pd
import plotnine as p9
import warnings


def volcano_plot(
            df, up_hit, down_hit,
            score_col='score', pvalue_col='pvalue',
            xlabel='phenotype score',
            ylabel=r'$-\log_{10}$(p-value)',
            ylabel_note=None,
            dot_size=1,
            xlims='auto',
            ylims='auto',
            ctrl_label='negative_control',
            resistance_hits=None,
            sensitivity_hits=None,
            size_txt=None,
            t_x=0, t_y=0,
            figure_size=(5, 5),
            ):
    """Draw a volcano plot with plotnine and return the `ggplot` object.

    `df` must have a `label` column (see `annotateScoreTable`). `resistance_hits` and
    `sensitivity_hits` are lists of targets to highlight (red / blue) and, if `size_txt`
    is set, to name with repelled text labels.
    """

    df = df.copy()
    y_col = f'-log10({pvalue_col})'
    if y_col not in df.columns:
        df[y_col] = -np.log10(df[pvalue_col])

    # limits come from finite values only (NaN/inf would make the axis limits invalid)
    x_fin = df[score_col][np.isfinite(df[score_col])]
    y_fin = df[y_col][np.isfinite(df[y_col])]
    if xlims == 'auto':
        xlims = (x_fin.min() - 0.1, x_fin.max() + 0.1)
    if ylims == 'auto':
        ylims = (y_fin.min() - 0.1, y_fin.max() + 0.1)

    colors = {
        'target_non_hit': 'black', up_hit: '#fcae91', down_hit: '#bdd7e7', ctrl_label: 'gray',
    }
    alphas = {'target_non_hit': 0.1, up_hit: 0.9, down_hit: 0.9, ctrl_label: 0.1}
    df['label'] = df['label'].astype(str)
    df = df[df['label'].isin(colors)]
    df['_alpha'] = df['label'].map(alphas)

    if ylabel_note:
        ylabel = f'{ylabel}\n({ylabel_note})'

    mapping = p9.aes(x=score_col, y=y_col)
    plot = (
        p9.ggplot(df, mapping)
        + p9.geom_point(p9.aes(color='label', alpha='_alpha'), size=dot_size, shape='o')
        + p9.scale_color_manual(values=colors, breaks=list(colors))
        + p9.scale_alpha_identity()
        + p9.coord_cartesian(xlim=xlims, ylim=ylims)
        + p9.labs(x=xlabel, y=ylabel, color='')
        + p9.theme_classic()
        + p9.theme(figure_size=figure_size)
    )

    label_frames = []
    for hits, hit_label, color in [
        (resistance_hits, up_hit, '#de2d26'),
        (sensitivity_hits, down_hit, '#3182bd'),
    ]:
        sub = _significant_target_rows(df, hits, hit_label)
        if sub is None: continue
        plot += p9.geom_point(sub, mapping, size=dot_size * 2, shape='o', color=color, alpha=1)
        label_frames.append(sub)

    if size_txt is not None and label_frames:
        txt = pd.concat(label_frames)
        plot += p9.geom_text(
            txt, p9.aes(label='target'), size=size_txt, color='black',
            nudge_x=t_x, nudge_y=t_y,
            adjust_text={'arrowprops': {'arrowstyle': '-', 'color': 'black', 'alpha': 0.2}},
        )

    return plot


def _significant_target_rows(df, targets, hit_label):
    """Rows of `targets` annotated as `hit_label`; warns about missing / non-significant
    targets and about skipped non-significant entries of multi-target genes."""
    if targets is None: return None
    if not isinstance(targets, (list, tuple, set)): targets = [targets]
    keep = []
    for target in targets:
        rows = df[df['target'] == target]
        if len(rows) == 0:
            warnings.warn(f'"{target}" not found in the results; it will not be labeled.')
            continue
        is_sig = rows['label'] == hit_label
        n_skipped = int((~is_sig).sum())
        if not is_sig.any():
            warnings.warn(f'"{target}" is not significant ({hit_label} not met); it will not be labeled.')
            continue
        if n_skipped:
            warnings.warn(
                f'"{target}" has {len(rows)} entries in the library; skipped {n_skipped} '
                f'that are not significant, labeling the other {int(is_sig.sum())}.')
        keep.append(rows[is_sig])
    return pd.concat(keep) if keep else None


# def drawRhoGammaScatter(
#         self, ax,
#         rho_df=None, gamma_df=None,
#         dot_size=1,
#         score_col='score',
#         xlabel='auto',
#         ylabel='auto',
#         xlims='auto',
#         ylims='auto',
#         **args
#         ):
#     #TODO: fix by making a single dataframe with both rho and gamma scores
#     if rho_df is None:
#         _, _, rho_df = self._prep_data()
#     if gamma_df is None:
#         gamma_df, _, _ = self._prep_data()

#     if xlabel == 'auto':
#         xlabel = self.rho_score_name.replace(':', ': ').replace('_', ' ')
#     if ylabel == 'auto':
#         ylabel = self.gamma_score_name.replace(':', ': ').replace('_', ' ')
    
#     # color by rho score labels
#     up_hit = 'resistance_hit'
#     down_hit = 'sensitivity_hit'

#     # Scatter plot for each category
#     ax.scatter( rho_df.loc[rho_df['label'] == 'target_non_hit', score_col],
#                 gamma_df.loc[rho_df['label'] == 'target_non_hit', score_col],
#                 alpha=0.1, s=dot_size, c='black', label='target_non_hit',
#                 **args)
    
#     ax.scatter( rho_df.loc[rho_df['label'] == up_hit, score_col],
#                 gamma_df.loc[rho_df['label'] == up_hit, score_col],
#                 alpha=0.9, s=dot_size, c='#fcae91', label=up_hit,
#                 **args)
    
#     ax.scatter( rho_df.loc[rho_df['label'] == down_hit, score_col],
#                 gamma_df.loc[rho_df['label'] == down_hit, score_col],
#                 alpha=0.9, s=dot_size, c='#bdd7e7', label=down_hit,
#                 **args)
    
#     ax.scatter( rho_df.loc[rho_df['label'] == self.ctrl_label, score_col],
#                 gamma_df.loc[rho_df['label'] == self.ctrl_label, score_col],
#                 alpha=0.1, s=dot_size, c='gray', label=self.ctrl_label,
#                 **args)
    
#     # Set x-axis and y-axis labels
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     # Set x-axis limits
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)

#     # Add legend
#     ax.legend()
    
