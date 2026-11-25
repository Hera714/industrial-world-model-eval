"""概率校准图：测试覆盖率、区间宽度与固定起点示例；不利用测试目标调参。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import plot_week5 as shared


def main():
    source=ROOT/'reports/week5/results/calibration'
    shared.OUT=source/'figures'; shared.OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    summary=pd.read_csv(source/'interval_metrics_by_horizon.csv')
    conditions=('recorded_observation','persistence_forecast')
    families=[('mlp','MLP','#D55E00'),('residual','Residual MLP','#009E73')]
    for metric,name,title,ylabel in [
        ('coverage_mean','coverage_by_horizon','Nominal 90% intervals: held-out test coverage','Coverage (%)'),
        ('mean_width_c','width_by_horizon','Nominal 90% intervals: prediction interval width','Mean interval width (°C, log scale)')]:
        fig,axes=shared.grid(title,ylabel)
        for i,condition in enumerate(conditions):
            for j,variable in enumerate(('air','wall')):
                ax=axes[i,j]
                for family,label,color in families:
                    for method,style in [('validation_calibrated','-'),('uncalibrated_gaussian_reference','--')]:
                        group=summary[(summary.family==family)&(summary.condition==condition)&
                            (summary['mode']=='free_running')&(summary.variable==variable)&(summary.method==method)].sort_values('horizon')
                        y=group[metric].to_numpy()*(100 if metric=='coverage_mean' else 1)
                        ax.plot(group.horizon,y,color=color,ls=style,lw=1.7,label=label if style=='-' else None)
                if metric=='coverage_mean':
                    ax.axhline(90,color='#555555',ls=':',lw=1)
                    ax.set_ylim(0,104); ax.set_yticks([0,25,50,75,90,100])
                else:
                    ax.set_yscale('log')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,.045))
        fig.text(.5,.015,'Solid: validation-calibrated. Dashed: Gaussian reference. 18 test episodes; no test-time fitting.',ha='center',fontsize=8)
        shared.save(fig,name)
    origins=pd.read_csv(source/'test/origins.csv')
    example_rows=[]
    for family,label,color in families:
        with np.load(source/'test'/f'{family}_intervals.npz') as data:
            fig,axes=shared.grid(f'{label}: pointwise 90% calibrated intervals at one fixed test origin','Temperature (°C)')
            for i,condition in enumerate(conditions):
                key=condition+'__free_running'
                for j,variable in enumerate(('air','wall')):
                    ax=axes[i,j]; h=np.arange(1,81)
                    actual=data['actual'][0,:,j]; mean=data[key+'__mean'][0,:,j]
                    low=data[key+'__validation_calibrated__lower'][0,:,j]
                    high=data[key+'__validation_calibrated__upper'][0,:,j]
                    base_low=data[key+'__uncalibrated_gaussian_reference__lower'][0,:,j]
                    base_high=data[key+'__uncalibrated_gaussian_reference__upper'][0,:,j]
                    ax.fill_between(h,low,high,color=color,alpha=.22,label='Calibrated interval')
                    ax.plot(h,base_low,color='#888888',ls='--',lw=.8,label='Gaussian reference bounds')
                    ax.plot(h,base_high,color='#888888',ls='--',lw=.8)
                    ax.plot(h,actual,color='black',lw=1.8,label='Simulator truth')
                    ax.plot(h,mean,color=color,lw=1.6,label='Ensemble mean')
                    example_rows.extend(dict(family=family,condition=condition,variable=variable,horizon=k+1,
                        actual=actual[k],mean=mean[k],lower=low[k],upper=high[k]) for k in range(80))
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.045),fontsize=8)
            fig.text(.5,.015,'Fixed first test origin; pointwise bands do not imply 90% simultaneous trajectory coverage.',ha='center',fontsize=8)
            shared.save(fig,f'{family}_interval_example')
    pd.DataFrame(example_rows).to_csv(shared.OUT/'interval_example_source.csv',index=False)
    (shared.OUT/'figure_notes.json').write_text(json.dumps(dict(origin=origins.iloc[0].to_dict(),
        scope='free-running; both conditions and both state variables; teacher-forcing metrics remain in CSV',
        nominal=.9,coverage='episode-equal mean; invalid intervals count as not covered',
        width='episode mean over finite valid intervals; see validity counts in tables',
        calibration='validation-only normalized residual quantiles; not a strict finite-sample guarantee'),indent=2))


if __name__=='__main__': main()
