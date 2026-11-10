"""导出已保存运行快照的命令行工具。"""
import argparse
import sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from shape_sensing.artifacts.packages import export_package

def main(argv=None):
    parser=argparse.ArgumentParser(description='从运行快照导出自包含算法参数包')
    parser.add_argument('--runs',nargs='+',required=True,help='一个或多个 runs/<sensor_id>/<run_id> 路径')
    parser.add_argument('--output',required=True,help='输出 JSON 文件')
    args=parser.parse_args(argv)
    export_package(args.runs,args.output)
    return 0
if __name__=='__main__': raise SystemExit(main())
