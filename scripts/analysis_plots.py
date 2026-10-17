"""Plot the numerical content of Figures 3-7 from analysis tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_core import ROOT, SEEDS, read

COLORS = ['#16858D', '#8066AB', '#C07E49']
FEATURES = ['frac_forest', 'slope_mean', 'clay_frac']
LABELS = ['Forest cover', 'Slope', 'Clay fraction']
COMPARISONS = ['M1-B0', 'M1-A1', 'A1-B0']


def write(fig, output, name):
    fig.savefig(output/(name+'.svg'), bbox_inches='tight')
    fig.savefig(output/(name+'.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def performance(data, out):
    runs, paired = read(data/'run_summary.csv'), read(data/'paired_nse.csv')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for model, color in zip(['B0', 'A1', 'M1'], COLORS):
        y = runs[runs.model == model].set_index('seed').loc[SEEDS].median_nse
        axes[0].plot(range(10), y, 'o-', label=model, color=color)
    axes[0].set_xticks(range(10), SEEDS, rotation=60)
    axes[0].set(xlabel='Training seed', ylabel='Median NSE')
    axes[0].legend(frameon=False)
    for comparison, color in zip(COMPARISONS, COLORS):
        p = paired[paired.comparison == comparison]
        values = p[[f'seed_{s}_delta' for s in SEEDS]].median(axis=0)
        axes[1].plot(range(10), values, 'o-', color=color, label=comparison)
        counts = p.improved_seeds.value_counts().reindex(range(11), fill_value=0)
        axes[2].plot(range(11), counts, 'o-', color=color, label=comparison)
    axes[1].set_xticks(range(10), SEEDS, rotation=60)
    axes[1].set(xlabel='Training seed', ylabel='Median paired ΔNSE')
    axes[2].set(xlabel='Seeds with positive ΔNSE', ylabel='Number of basins', xticks=range(11))
    axes[2].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color='.6', lw=.7)
    write(fig, out, 'figure3_performance')


def groups(data, out):
    table = read(data/'attribute_group_summary.csv')
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for j, (feature, label) in enumerate(zip(FEATURES, LABELS)):
        for i, comparison in enumerate(COMPARISONS[:2]):
            ax = axes[i, j]
            for k, state in enumerate(['mid', 'high']):
                p = table[(table.feature == feature) & (table.comparison == comparison) &
                          (table.metric == 'mae') & (table.state == state)].set_index('level').loc[['low', 'middle', 'high']]
                ax.plot(range(3), p.median_seedmean_delta, 'o-', color=COLORS[k],
                        label='Medium flow' if state == 'mid' else 'High flow')
            ax.set_xticks(range(3), ['Low', 'Medium', 'High'])
            ax.axhline(0, color='.6', lw=.7)
            ax.set_title(f'{label}: {comparison}')
            ax.set_ylabel('Median ΔMAE (mm/day)')
    axes[0, 0].legend(frameon=False)
    write(fig, out, 'figure4_environmental_groups')


def gates(data, out):
    density = read(data/'gates/gate_density_data.csv')
    scatter = read(data/'gates/gate_error_scatter.csv')
    values = density.high_minus_mid_pp.to_numpy()
    bandwidth = .9*min(values.std(ddof=1), (np.quantile(values, .75)-np.quantile(values, .25))/1.34)*len(values)**(-.2)
    grid = np.linspace(values.min()-3*bandwidth, values.max()+3*bandwidth, 400)
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), constrained_layout=True)
    for i, (feature, label) in enumerate(zip(FEATURES, LABELS)):
        for level, color in zip(['low', 'middle', 'high'], COLORS):
            v = density.loc[density[feature+'_group'] == level, 'high_minus_mid_pp'].to_numpy()
            curve = np.exp(-.5*((grid[:, None]-v)/bandwidth)**2).mean(axis=1)/(bandwidth*np.sqrt(2*np.pi))
            axes[i, 0].plot(grid, curve, color=color, label=level.capitalize().replace('Middle', 'Medium'))
        axes[i, 0].set(title=label, xlabel='High–medium net gate modulation (pp)', ylabel='Density')
        axes[i, 0].legend(frameon=False, fontsize=8)
        state = ['low', 'mid', 'high'][i]
        for comparison, color in zip(COMPARISONS[:2], COLORS):
            p = scatter[(scatter.state == state) & (scatter.comparison == comparison)]
            axes[i, 1].scatter(p.net_gate_shift*100, p.mae_change, s=12, color=color, label=comparison, alpha=.7)
        axes[i, 1].set(title={'low':'Low flow', 'mid':'Medium flow', 'high':'High flow'}[state],
            xlabel='Net gate modulation (pp)', ylabel='ΔMAE (mm/day)')
        axes[i, 1].axhline(0, color='.6', lw=.7)
    axes[0, 1].legend(frameon=False, fontsize=8)
    write(fig, out, 'figure5_gate_modulation')


def rainfall(data, out):
    curves = read(data/'gates/basin_seed_bin_curves.csv')
    summary = read(data/'gates/rainfall_group_summary.csv')
    attrs = read(data/'basin_attributes_and_groups.csv')
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True, sharex=True, sharey=True)
    for col, (group, feature, label) in enumerate(zip(['high_forest', 'high_slope', 'high_clay'], FEATURES, LABELS)):
        members = attrs.loc[attrs[feature+'_group'] == 'high', 'basin_id']
        axes[0, col].set_title('High '+label.lower())
        for row, predictor in enumerate(['p3', 'ap30']):
            ax = axes[row, col]
            for index, (flow, color) in enumerate(zip(['mid', 'high'], COLORS)):
                p = curves[curves.basin_id.isin(members) & (curves.predictor == predictor) & (curves.flow == flow)]
                matrix = p.groupby(['basin_id', 'bin']).mean_adjusted_gate.mean().unstack().to_numpy()
                x = np.arange(10, 100, 20)
                for tail in np.linspace(0, .45, 61):
                    ax.fill_between(x, np.quantile(matrix, tail, axis=0), np.quantile(matrix, 1-tail, axis=0),
                                    color=color, alpha=.025, linewidth=0)
                ax.plot(x, matrix.mean(axis=0), color=color, label='Medium flow' if flow == 'mid' else 'High flow')
                s = summary[(summary.group == group) & (summary.flow == flow) & (summary.predictor == predictor)].iloc[0]
                ax.text(.03, .97-index*.09, f'{"Medium" if flow == "mid" else "High"}: ρ = {s["mean"]:.3f}, q = {s.q_fdr:.3f}',
                        transform=ax.transAxes, va='top', fontsize=8)
            ax.axhline(0, lw=.6, color='.6')
            ax.set_ylim(-1.25, 1.7)
            ax.set_xticks(range(10, 100, 20))
            if col == 0:
                ax.set_ylabel(('P3' if row == 0 else 'AP30')+'\nNet gate modulation\n(standardized rank residual)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.015), ncol=2, frameon=False)
    fig.supxlabel('Within-basin percentile of adjusted precipitation (%)')
    write(fig, out, 'figure6_precipitation')


def temporal(data, out):
    table = read(data/'temporal/cross_period_basin_effects.csv')
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for row, comparison in enumerate(['M1_B0', 'M1_A1']):
        for col, flow in enumerate(['low', 'mid', 'high']):
            p = table[(table.threshold_scheme == 'period_relative') & (table.comparison == comparison) &
                      (table.flow == flow) & (table.metric == 'delta_mae')]
            x, y = p.original_delta.to_numpy(), p.confirmation_delta.to_numpy()
            colors = np.where((x < 0) & (y < 0), COLORS[0], np.where((x > 0) & (y > 0), '#BA5B4E', COLORS[1]))
            ax = axes[row, col]
            ax.scatter(x, y, c=colors, s=15, alpha=.8)
            ax.axhline(0, color='.6', lw=.7)
            ax.axvline(0, color='.6', lw=.7)
            ax.set_xscale('symlog', linthresh=.02)
            ax.set_yscale('symlog', linthresh=.02)
            ax.set_title(comparison.replace('_', '−')+'; '+{'low':'Low', 'mid':'Medium', 'high':'High'}[flow]+' flow')
            ax.set(xlabel='2003–2008 ΔMAE (mm/day)', ylabel='1989–1999 ΔMAE (mm/day)')
    write(fig, out, 'figure7_cross_period')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT/'runs/analysis')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    for plot in [performance, groups, gates, rainfall, temporal]:
        plot(args.analysis, args.output)
    print(f'Wrote five SVG and 600 dpi PNG figure pairs to {args.output}')


if __name__ == '__main__':
    main()
