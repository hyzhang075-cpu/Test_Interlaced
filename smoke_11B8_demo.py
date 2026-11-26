import os,sys,json
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')]
os.chdir(root/'new_codebase');os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
from PyQt5.QtWidgets import QApplication
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.runtime.demo import LiveFrameProcessor
from shape_sensing.runtime.source import InterrogatorSource
from shape_sensing.input_preprocessing._kernel import InputKernel
from unittest.mock import patch
package='runs/11B8_core_point_refine_k1_end05/11B8_validated_algorithm.json'
args=parse_main_args(['--sensor_params',package,'--task','demo'])
configure_calibration_args(args)
session=TaskSession(module_parameters(args,optimizer_parameters),task='demo',region='full')
from shape_sensing.runtime.service import RuntimeService
runtime=RuntimeService(LiveFrameProcessor(session,args),source=InterrogatorSource(session.config.input),direct=240)
channels=list(session.config.input.active_channels)
absolute=np.full((20,7),1550.)+np.arange(140).reshape(20,7)*.0001
delta=np.tile(np.arange(7),(20,1))*.00001
import struct
packets=[]
for values in (absolute,absolute+delta):
 body=json.dumps([[100*ch+i+1,float(values[i,j])] for j,ch in enumerate(channels) for i in range(20)]).encode()
 packets.append(struct.pack('<I',len(body))+body)
class Wire:
 def __init__(self):self.packets=iter(packets)
 def settimeout(self,*a):pass
 def connect(self,*a):pass
 def recv(self,*a):return next(self.packets,b'')
 def shutdown(self,*a):pass
 def close(self):pass
with patch('shape_sensing.runtime.source.socket.socket',return_value=Wire()),patch.object(InputKernel,'_parse_packet',lambda self,b:json.loads(b)),patch('shape_sensing.runtime.source.time.monotonic',side_effect=[1.,2.]):
 source=runtime.source;source.open()
 for _ in range(2):
  merged=source.read();assert merged.shape==(40,4)
  frame=runtime.process_frame(merged)
  assert frame.shape.position_m.shape==(256,3)
  assert np.isfinite(frame.shape.position_m).all()
 source.close()
runtime.close()
print('11B8 demo: package + simulated 20x7 packets -> 40x4 responses -> 256x3 shape: PASS',flush=True)
