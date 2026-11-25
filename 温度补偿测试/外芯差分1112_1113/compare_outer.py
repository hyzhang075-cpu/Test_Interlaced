"""Offline center versus outer differences; production orthogonal HT integration."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HELPER=Path(r'C:\Users\win11\Desktop\project\1.算法开发\4.现有项目优化\改为RK45\shape-sensing-integrate-08B5\shape-sensing-integrate-08B5\数据分析')
sys.path.insert(0,str(HELPER))
import replay_08B5_orthogonal_temperature as h
OUT=Path(__file__).resolve().parent

def main():
    recon=h.make_reconstructor()
    original_inverse=recon.core_inverse.copy()
    positions=np.linalg.pinv(original_inverse)
    D=np.array([[1,-1,0],[1,0,-1],[0,1,-1.]])
    # All pair differences have rank two. This composite equals GLS on
    # [23,24] with covariance sigma² [[2,1],[1,2]], for equal outer noise.
    outer_inverse=np.linalg.pinv(D@positions)@D
    D2=D[:2]; covariance=D2@D2.T
    Q=D2@positions
    gls=np.linalg.inv(Q.transpose(0,2,1)@np.linalg.inv(covariance)@Q)@Q.transpose(0,2,1)@np.linalg.inv(covariance)@D2
    assert np.allclose(gls,outer_inverse)
    assert np.max(abs(outer_inverse@np.ones(3)))<1e-8
    assert np.allclose(outer_inverse@positions,np.eye(2))
    coeff=pd.read_csv(h.COEFF)
    k=np.array([[coeff[(coeff.FBG==i)&(coeff.channel==ch)].heating_pm_per_C.iloc[0] for ch in range(1,5)] for i in range(1,22)])
    ratio=k[:,1:]/k[:,:1]
    summary=[]
    for name,info in h.CASES.items():
        cache=OUT/(name+'_bins.npz')
        if cache.exists():
            z=np.load(cache); t0=float(z['t0']); times=z['times']; raw=z['raw']
        else:
            t0,first,rows=h.read_binned(h.DATA/info['file'])
            times=np.array([r[0] for r in rows]); raw=np.stack([r[1] for r in rows])
            np.savez_compressed(cache,t0=t0,times=times,raw=raw)
        low=h.mask_window(name,info,times,t0,'low'); high=h.mask_window(name,info,times,t0,'high')
        assert low.sum()>20 and high.sum()>20
        center=raw[:,:,1:]-raw[:,:,:1]
        point=center-(ratio[None]-1)*(raw[:,:,0]-raw[low,:,0].mean(0))[:,:,None]
        methods={'center':(center,original_inverse),'outer':(raw[:,:,1:],outer_inverse),'pointwise_legacy':(point,original_inverse)}
        groups=((times-t0)//300).astype(int); table=[]; bases={}; highs={}
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for label,(values,inverse) in methods.items():
            recon.core_inverse=inverse
            base=h.shape_from_outer(recon,values[low].mean(0)); bases[label]=base
            highshape=h.shape_from_outer(recon,values[high].mean(0)); highs[label]=highshape
            dist=np.linalg.norm(highshape-base,axis=1)
            delta=(values[high].mean(0)-values[low].mean(0))*1e6
            if label=='outer': delta=delta@D.T
            summary.append(dict(case=name,method=label,tip_mm=dist[-1],whole_RMS_mm=np.sqrt(np.mean(dist**2)),max_mm=dist.max(),wave_RMS_pm=np.sqrt(np.mean(delta**2)),low_bins=int(low.sum()),high_bins=int(high.sum())))
            ys=[]; xs=[]
            for g in np.unique(groups):
                m=groups==g; curve=h.shape_from_outer(recon,values[m].mean(0)); diff=np.linalg.norm(curve-base,axis=1)
                hour=(times[m].mean()-t0)/3600
                table.append(dict(method=label,hour=hour,tip_mm=diff[-1],whole_RMS_mm=np.sqrt(np.mean(diff**2))))
                xs.append(hour); ys.append(np.sqrt(np.mean(diff**2)))
            axes[0].plot(xs,ys,label=label)
            axes[1].plot(np.linspace(0,895,len(dist)),dist,label=label)
        axes[0].set(xlabel='Hours from recording start',ylabel='Whole-shape RMS drift / mm')
        axes[1].set(xlabel='Arc length / mm',ylabel='High minus low position drift / mm')
        for ax in axes: ax.grid(alpha=.25); ax.legend()
        fig.suptitle(name+' | orthogonal + HT | each method uses own low baseline')
        fig.tight_layout(); fig.savefig(OUT/(name+'.png'),dpi=160); plt.close(fig)
        pd.DataFrame(table).to_csv(OUT/(name+'_time.csv'),index=False)
        np.savez_compressed(OUT/(name+'_plateau_shapes.npz'),**{f'{a}_{b}':v for a,d in [('low',bases),('high',highs)] for b,v in d.items()})
        print(pd.DataFrame(summary).tail(3).to_string(index=False),flush=True)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    (OUT/'verification.json').write_text(json.dumps({'outer_GLS_equivalence':True,'geometry_identity':True,'max_pair_condition':float(np.linalg.cond(Q).max()),'calibration':str(h.PKL),'data':str(h.DATA)},indent=2),encoding='utf-8')

if __name__=='__main__': main()
