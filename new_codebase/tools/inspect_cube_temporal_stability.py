"""Read-only temporal stability diagnostics; never changes calibration data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
folder=root/'1124-11B1+导管0状态＋定标测试/全段测试'
rows=[]
for direct in (0,120,240):
    for radius in (300,600,1000,2000):
        for path in (folder/f'{direct}-{radius}').glob('Channel_*.csv'):
            frame=pd.read_csv(path)
            values=frame.iloc[:,2:20].apply(pd.to_numeric,errors='coerce').to_numpy()
            values=values[np.isfinite(values).all(axis=1)&(np.abs(values)<9000).all(axis=1)]
            n=len(values)
            if n<20:continue
            delta=values[-n//4:].mean(axis=0)-values[:n//4].mean(axis=0)
            row={'direct':direct,'radius':radius,'channel':int(path.name.split('_')[1]),'rows':n,
                 'first_last_quarter_shift_rms_pm':float(np.sqrt(np.mean(delta**2))),
                 'first_last_quarter_shift_max_pm':float(np.abs(delta).max()),
                 'mean_temporal_std_pm':float(np.std(values,axis=0).mean()),
                 'wl1_shift_pm':float(delta[0]),'wl2_shift_pm':float(delta[1]),'wl3_shift_pm':float(delta[2])}
            rows.append(row)
out=root/'runs/11B1_Cube/temporal_stability.json'
out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
for row in sorted(rows,key=lambda r:r['first_last_quarter_shift_rms_pm'],reverse=True)[:10]:print(row)
