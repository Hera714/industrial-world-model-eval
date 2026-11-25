"""绘制 Week 5 长时序评测：误差累积、信息边界与固定起点预测；不重新训练模型。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'reports/week5/results/rollout'
OUT = ROOT/'reports/week5/figures'
FAMILIES = [('persistence', 'Persistence', '#777777'), ('linear', 'Linear', '#0072B2'),
            ('action_mlp', 'MLP', '#D55E00'), ('residual_mlp', 'Residual MLP', '#009E73')]


def save(fig, name):
    """输出可编辑 PDF 和预览 PNG，并检查最终面板对齐。"""
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=OUT/f'{name}.alignment.json', strict=True)
    fig.savefig(OUT/f'{name}.png', dpi=300)
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)


def grid(title, ylabel):
    """相同变量放在同一列，天气信息条件分行，固定布局防止标签挤压。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.88, hspace=.45, wspace=.28)
    fig.suptitle(title, fontsize=14, y=.97)
    for i, condition in enumerate(('Recorded outdoor observations', 'Outdoor held at origin')):
        for j, variable in enumerate(('Air temperature', 'Wall temperature')):
            ax = axes[i, j]
            ax.set_title(f'{condition} / {variable}', fontsize=10)
            ax.set_xlabel('Prediction horizon (10-min steps)')
            ax.set_ylabel(ylabel)
            ax.set_xticks([1, 20, 40, 60, 80])
            ax.grid(alpha=.2)
    return fig, axes


def main(config_path=None):
    """使用所有 episode 等权汇总；阴影仅表示三个训练种子间的样本标准差。"""
    global DATA, OUT
    if config_path is not None:
        import yaml
        config=yaml.safe_load(Path(config_path).read_text())
        if config['max_horizon']!=80:
            raise ValueError('These frozen-protocol figures require max_horizon=80')
        DATA=ROOT/config['output']/'rollout'
        OUT=(ROOT/config['output']).parent/'figures'
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'svg.fonttype': 'none', 'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False})
    metrics = pd.read_csv(DATA/'metrics_across_seeds.csv')
    key = metrics[metrics.horizon.isin([1, 5, 10, 20, 40, 80])]
    key.to_csv(OUT/'key_horizon_metrics.csv', index=False)
    for definition, filename, title in [('terminal', 'terminal_rmse', 'Free-running: error accumulated at each horizon'),
                                        ('prefix_complete', 'prefix_rmse', 'Free-running: error averaged over steps 1 to H')]:
        fig, axes = grid(title, 'RMSE (°C, log scale)')
        for i, condition in enumerate(('recorded_observation', 'persistence_forecast')):
            for j, variable in enumerate(('air', 'wall')):
                ax = axes[i, j]
                for family, label, color in FAMILIES:
                    group = metrics[(metrics.family == family) & (metrics['mode'] == 'free_running') &
                        (metrics.condition == condition) & (metrics.variable == variable) &
                        (metrics.definition == definition)].sort_values('horizon')
                    x, y, sd = (group[c].to_numpy() for c in ('horizon', 'rmse_c_mean', 'rmse_c_seed_sd'))
                    ax.plot(x, y, color=color, label=label, lw=1.8)
                    if np.isfinite(sd).any():
                        ax.fill_between(x, np.maximum(y-sd, 1e-6), y+sd, color=color, alpha=.13)
                ax.set_yscale('log')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=4, bbox_to_anchor=(.5, .045))
        fig.text(.5, .015, '18 episodes; lines: seed means; bands: ±1 training-seed SD (MLPs only).', ha='center', fontsize=9)
        save(fig, filename)

    # Teacher forcing 对照：同一模型、同一信息条件，实线递推与虚线真实状态输入。
    fig, axes = grid('State feedback reveals the gap between one-step and recursive prediction', 'Terminal RMSE (°C, log scale)')
    for i, condition in enumerate(('recorded_observation', 'persistence_forecast')):
        for j, variable in enumerate(('air', 'wall')):
            for family, label, color in FAMILIES:
                for mode, style in [('free_running', '-'), ('teacher_forcing', '--')]:
                    g = metrics[(metrics.family == family) & (metrics['mode'] == mode) &
                        (metrics.condition == condition) & (metrics.variable == variable) & (metrics.definition == 'terminal')].sort_values('horizon')
                    axes[i, j].plot(g.horizon, g.rmse_c_mean, color=color, ls=style, lw=1.5, label=label if mode == 'free_running' else None)
            axes[i, j].set_yscale('log')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=4, bbox_to_anchor=(.5, .045))
    fig.text(.5, .015, 'Solid: free-running. Dashed: teacher forcing (true current state supplied at every step).', ha='center', fontsize=9)
    save(fig, 'teacher_vs_free')

    # 预先固定第一个测试 episode 的第一个起点，不按模型表现挑选样例。
    origins = pd.read_csv(DATA/'origins.csv')
    samples = {}
    for family, label, color in FAMILIES:
        names = [family] if family in ('linear', 'persistence') else [f'{"mlp" if family == "action_mlp" else "residual"}_seed{s}' for s in (42, 43, 44)]
        samples[family] = [np.load(DATA/f'{name}_predictions.npz') for name in names]
    fig, axes = grid('Example rollout: fixed first test episode, source step 0', 'Temperature (°C)')
    source_rows = []
    for i, condition in enumerate(('recorded_observation', 'persistence_forecast')):
        for j, variable in enumerate(('air', 'wall')):
            actual = samples['linear'][0]['actual'][0, :, j]
            axes[i, j].plot(np.arange(1, 81), actual, color='black', lw=2, label='Simulator truth')
            for family, label, color in FAMILIES:
                values = np.stack([a[condition+'__free_running'][0, :, j] for a in samples[family]])
                mean = values.mean(axis=0)
                axes[i, j].plot(np.arange(1, 81), mean, color=color, label=label, lw=1.6)
                if len(values) > 1:
                    sd = values.std(axis=0, ddof=1)
                    axes[i, j].fill_between(np.arange(1, 81), mean-sd, mean+sd, color=color, alpha=.13)
                source_rows.extend(dict(condition=condition, variable=variable, family=family, horizon=k+1,
                    actual=actual[k], predicted_mean=mean[k]) for k in range(80))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=5, bbox_to_anchor=(.5, .045), fontsize=9)
    fig.text(.5, .015, 'One fixed origin; recorded actions; bands show model-seed spread, not 90% prediction intervals.', ha='center', fontsize=9)
    save(fig, 'rollout_example')
    pd.DataFrame(source_rows).to_csv(OUT/'example_source_data.csv', index=False)
    (OUT/'figure_notes.json').write_text(json.dumps(dict(origin=origins.iloc[0].to_dict(),
        aggregation='episode equal-weight mean, then seed mean', spread='sample SD across 3 training seeds; not CI',
        curve_scale='RMSE figures use logarithmic y axes; no observations removed',
        source=str(DATA), conclusion='Quantify error accumulation and distinguish provided-weather from origin-only forecasting'), indent=2))
    print(key.query('mode == "free_running" and condition == "recorded_observation" and definition == "terminal" and horizon in [1,5,10,20,80]')[['family','variable','horizon','mae_c_mean','rmse_c_mean']].to_string(index=False))
    plot_evaluation_diagnostics(metrics)


def plot_evaluation_diagnostics(metrics):
    """补充有效率、回合工况分布和逐 horizon 排名；所有图导出对应源数据表。"""
    selected=metrics[(metrics['mode']=='free_running') & (metrics.definition=='terminal')].copy()
    selected.to_csv(OUT/'ranking_and_validity_source.csv',index=False)
    labels=[label for _,label,_ in FAMILIES]
    families=[family for family,_,_ in FAMILIES]
    conditions=('recorded_observation','persistence_forecast')
    # 同为100%时曲线会完全重叠，改用固定0–100%色阶展示每个模型单独的有效率。
    fig,axes=grid('Finite prediction availability: every requested origin remains in the denominator','Model')
    fig.subplots_adjust(left=.19,wspace=.38)
    for i,condition in enumerate(conditions):
        for j,variable in enumerate(('air','wall')):
            group=selected[(selected.condition==condition)&(selected.variable==variable)]
            matrix=group.pivot(index='family',columns='horizon',values='valid_fraction_mean').reindex(families)
            ax=axes[i,j]
            im=ax.imshow(matrix.to_numpy(),aspect='auto',origin='upper',extent=[.5,80.5,3.5,-.5],
                         cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
            ax.set_yticks(range(4),labels); ax.grid(False)
            for row in range(4):
                values=matrix.iloc[row].to_numpy()
                if np.isfinite(values).all() and np.allclose(values,values[0]):
                    ax.text(40,row,f'{values[0]:.0%} at all horizons',ha='center',va='center',
                            color='white' if values[0]>.6 else 'black',fontsize=9)
    fig.text(.5,.075,'Colour scale: white = 0%, dark blue = 100%. MLP values averaged across 3 fixed training seeds.',ha='center',fontsize=9)
    fig.text(.5,.03,'Finite availability is not physical validity; rate/boundary warnings may still occur.',ha='center',fontsize=9)
    save(fig,'valid_prediction_fraction')

    # 排名只使用无停止失败且完整支持的模型族，不用幸存子样本的低误差抬高排名。
    selected['rank_eligible']=(selected.failed_model_origins==0)&(selected.valid_fraction_mean==1)&(selected.runs_with_valid_error==selected.runs)
    selected['rank']=np.nan
    valid=selected[selected.rank_eligible]
    selected.loc[valid.index,'rank']=valid.groupby(['condition','horizon','variable']).rmse_c_mean.rank(method='min')
    selected.to_csv(OUT/'model_ranking_by_horizon.csv',index=False)
    fig,axes=grid('Model ordering depends on forecast horizon and available weather information','RMSE rank (1 = lowest error)')
    for i,condition in enumerate(conditions):
        for j,variable in enumerate(('air','wall')):
            ax=axes[i,j]
            for family,label,color in FAMILIES:
                group=selected[(selected.family==family)&(selected.condition==condition)&(selected.variable==variable)].sort_values('horizon')
                ax.plot(group.horizon,group['rank'],color=color,lw=1.8,label=label)
            ax.set_yticks([1,2,3,4]); ax.set_ylim(4.25,.75)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.045))
    fig.text(.5,.015,'Ranks use seed-mean episode RMSE; exact ties share the minimum rank. No significance claim.',ha='center',fontsize=9)
    save(fig,'ranking_by_horizon')

    episodes=pd.read_csv(DATA/'metrics_by_episode_across_seeds.csv')
    origins=pd.read_csv(DATA/'origins.csv')
    metadata=origins[['episode_id','policy','initial_air_c']].drop_duplicates()
    if metadata.episode_id.duplicated().any(): raise ValueError('Episode metadata is not unique')
    episodes=episodes[(episodes['mode']=='free_running')&(episodes.definition=='terminal')&(episodes.horizon==80)].merge(metadata,on='episode_id',validate='many_to_one')
    episodes['initial_air_bin']=pd.cut(episodes.initial_air_c,[-np.inf,19,21,np.inf],right=False,
                                     labels=['<19 C','19–21 C','>=21 C']).astype(str)
    episodes.to_csv(OUT/'condition_distribution_source.csv',index=False)
    # 分别按策略和初温展示回合分布，所有18个回合都保留；箱体是回合分布而非种子分布。
    from matplotlib.patches import Patch
    for column,name in [('policy','policy'),('initial_air_bin','initial_temperature')]:
        categories=sorted(metadata.policy.unique()) if column=='policy' else ['<19 C','19–21 C','>=21 C']
        fig,axes=grid(f'H=80: episode error distribution by {"collection policy" if column=="policy" else "initial air temperature"}','Terminal RMSE (°C, log scale)')
        for i,condition in enumerate(conditions):
            for j,variable in enumerate(('air','wall')):
                ax=axes[i,j]
                frame=episodes[(episodes.condition==condition)&(episodes.variable==variable)]
                ticklabels=[]
                for position,category in enumerate(categories):
                    part=frame[frame[column]==category]
                    n=part.episode_id.nunique()
                    ticklabels.append(f'{category}\n(n={n} episodes)')
                    for m,(family,label,color) in enumerate(FAMILIES):
                        values=part[part.family==family].sort_values('episode_id').rmse_c_mean.to_numpy()
                        if not np.isfinite(values).all():
                            raise ValueError('Missing episode error: disclose incomplete groups rather than silently dropping values')
                        if not len(values): continue
                        center=position+(m-1.5)*.19
                        box=ax.boxplot([values],positions=[center],widths=.15,patch_artist=True,
                            whis=(0,100),showfliers=False,manage_ticks=False,
                            medianprops=dict(color=color,lw=1.5))
                        box['boxes'][0].set(facecolor=color,alpha=.18,edgecolor=color)
                        ax.scatter(center+np.linspace(-.035,.035,len(values)),values,s=10,color=color,alpha=.8,zorder=3)
                ax.set_xticks(range(len(categories)),ticklabels,fontsize=8)
                ax.set_xlim(-.6,len(categories)-.4); ax.set_xlabel('Episode group')
                ax.set_yscale('log')
        fig.legend(handles=[Patch(facecolor=color,alpha=.5,label=label) for _,label,color in FAMILIES],
                   loc='lower center',ncol=4,bbox_to_anchor=(.5,.045))
        fig.text(.5,.015,'Each dot = one episode after seed averaging. Box: Q1/median/Q3; whiskers: min/max. All episodes retained.',ha='center',fontsize=8)
        save(fig,'condition_distribution_'+name)

    # 为报告生成按条件明确分开的关键步长表，避免复制粘贴时混淆两种天气信息集。
    table=selected[selected.horizon.isin([1,5,10,20,40,80])].copy()
    table.to_csv(OUT/'evaluation_tables.csv',index=False)
    lines=['# Week 5 关键结果表','',
           '误差先按 episode 等权，再对固定训练种子平均。± 为训练种子样本标准差；确定性基线不虚构种子 SD。',
           '有效率为有限预测可用率；排名仅为数值排序，不代表显著差异。','']
    for condition in conditions:
        for variable in ('air','wall'):
            lines.extend([f'## {condition} / {variable} / free_running terminal','',
                '| Model | H | MAE (°C) | RMSE (°C) | Finite availability | Rank |',
                '|---|---:|---:|---:|---:|---:|'])
            for family,label,_ in FAMILIES:
                group=table[(table.family==family)&(table.condition==condition)&(table.variable==variable)].sort_values('horizon')
                for row in group.itertuples():
                    def display(mean,sd):
                        return f'{mean:.5f} ± {sd:.5f}' if np.isfinite(sd) else f'{mean:.5f}'
                    lines.append(f'| {label} | {row.horizon} | {display(row.mae_c_mean,row.mae_c_seed_sd)} | {display(row.rmse_c_mean,row.rmse_c_seed_sd)} | {row.valid_fraction_mean:.1%} | {row.rank:g} |')
            lines.append('')
    (OUT/'evaluation_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'diagnostic_figure_notes.json').write_text(json.dumps(dict(
        scope='Frozen simple_thermal main evaluation; IB and projected models remain separate extensions',
        requested_episodes=int(metadata.episode_id.nunique()),
        policy_episode_counts=metadata.groupby('policy').size().to_dict(),
        condition_horizon=80,distribution_unit='episode; fixed training seeds averaged within episode',
        distribution_box='Q1/median/Q3, min/max whiskers, every episode shown as a point',
        ranking='terminal RMSE seed mean; min-rank exact ties; no ranking of incomplete/failed supports',
        availability='finite available prediction / all requested origins; not physical compliance',
        uncertainty='error curve shading is training-seed sample SD; box spread is episode variation',
        exports='300 dpi PNG, editable PDF/SVG, alignment reports and source CSV tables'),indent=2))


if __name__ == '__main__':
    main()
