"""Run the separate leakage-controlled V2 evaluation; never overwrites Step09."""
from pathlib import Path
import argparse,os,sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spm_v2.leakage_evaluation import run

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--teacher',required=True);p.add_argument('--raw-features',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);p.add_argument('--benchmark',action='store_true')
    p.add_argument('--feature-classes',required=True,help='Frozen reviewed TSV: feature_name, feature_class (or strict_class), reason; exact raw feature pool')
    p.add_argument('--reuse-spatial-from',help='Optional completed earlier grouped run; reuse B only after exact computational/dependency verification')
    run(p.parse_args())
