"""经验 CRPS 曲线：只评分原始三成员经验分布，不把区间校准误标成完整分布校准。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import plot_week5 as shared


def main():
    source=ROOT/'reports/week5/results/probability_scores'
    shared.OUT=source/'figures'; shared.OUT.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(source/'scores_by_horizon.csv')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=shared.grid('Empirical CRPS of the original equal-weight 3-member distribution','Empirical CRPS (°C)')
    for i,condition in enumerate(('recorded_observation','persistence_forecast')):
        for j,variable in enumerate(('air','wall')):
            ax=axes[i,j]
            values=frame[(frame.condition==condition)&(frame.variable==variable)&(frame.definition=='terminal')]
            for family,label,color in [('mlp','MLP','#D55E00'),('residual','Residual MLP','#009E73')]:
                for mode,style in [('free_running','-'),('teacher_forcing','--')]:
                    group=values[(values.family==family)&(values['mode']==mode)].sort_values('horizon')
                    ax.plot(group.horizon,group.empirical_crps_c_mean,color=color,ls=style,lw=1.7,
                            label=label if mode=='free_running' else None)
            # 零分数合法，不为对数坐标删掉完美预测；出现0时本面板保持线性。
            if (values.empirical_crps_c_mean>0).all():
                ax.set_yscale('log'); ax.set_ylabel('Empirical CRPS (°C, log scale)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,.045))
    fig.text(.5,.015,'Solid: free-running. Dashed: teacher forcing. Episode-equal means; lower is better. No density/NLL assumed.',ha='center',fontsize=8)
    shared.save(fig,'empirical_crps_by_horizon')
    frame.to_csv(shared.OUT/'crps_curve_source.csv',index=False)


if __name__=='__main__': main()
