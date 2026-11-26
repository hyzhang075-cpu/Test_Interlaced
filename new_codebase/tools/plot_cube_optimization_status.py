"""Plot separately verified candidates without implying a passing combined set."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];run=root/'runs/11B1_Cube'
single=json.loads((run/'pointwise_core_refined2_linear_levels10_kend05/full.json').read_text())['metrics']
joint=json.loads((run/'pointwise_core_fusion_joint_low_kend05/fusion_gain_refinement.json').read_text())['metrics']
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,title,data in [(axes[0],'Single-region candidate (separate fit)',single),(axes[1],'Joint candidate: actual full fusion',{d:m['full'] for d,m in joint.items()})]:
    x=np.arange(3)
    for shift,key,label,color in [(-.18,'point','Mean point','#2876bd'),(.18,'end','Endpoint','#e79428')]:
        values=[100*data[str(d)][key] for d in (0,120,240)]
        bars=ax.bar(x+shift,values,width=.34,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.axhline(1,color='#b42727',linestyle='--',label='1% target')
    ax.set_xticks(x,['0 deg','120 deg','240 deg']);ax.set_title(title,fontsize=10)
    ax.set_ylim(0,2.8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Error (%)');axes[0].legend(fontsize=8)
fig.suptitle('11B1_Cube | 256 points | k_point=1, k_end=0.5',fontsize=12)
fig.text(.5,.015,'Separately verified candidates; no fully passing combined parameter set yet.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.045,1,.94]);fig.savefig(run/'optimization_status.png',dpi=160)
print(run/'optimization_status.png')
