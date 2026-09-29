from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'source'/'governed_evidence_database_151.csv'
TAB=BASE/'tables'; TAB.mkdir(exist_ok=True)
THRESHOLD=3076.0

def main():
    df=pd.read_csv(SRC,keep_default_na=False)
    if len(df)!=151:
        raise ValueError(f'Expected 151 configurations, found {len(df)}')
    if df['RecordID'].duplicated().any():
        raise ValueError('Duplicate RecordID detected')
    if (df['PaperGroupID'].astype(str).str.len()==0).any():
        raise ValueError('Missing PaperGroupID detected')
    df['Support']=df['Support'].replace({'':'None'})
    df['Modifier']=df['Modifier'].replace({'':'None'})
    df['HierarchicalFlag']=(df['Morphology']=='Nanoflower').astype(int)
    df['ChemMorph']=df['BroadChemistry']+'|'+df['Morphology']
    df['ChemSupport']=df['BroadChemistry']+'|'+df['Support']
    df['MorphSupport']=df['Morphology']+'|'+df['Support']
    df['LogSensitivity']=np.log10(1+df['Sensitivity_uA_mM_cm2'].astype(float))
    df['HighPerformanceLabel']=(df['Sensitivity_uA_mM_cm2'].astype(float)>=THRESHOLD).astype(int)
    out=TAB/'final_database_151_v5.csv'
    df.to_csv(out,index=False)
    qa={
        'version':'v5.0 PARETO-LOCKED',
        'n_rows':int(len(df)),
        'n_paper_groups':int(df.PaperGroupID.nunique()),
        'duplicate_record_ids':int(df.RecordID.duplicated().sum()),
        'missing_sensitivity':int(df.Sensitivity_uA_mM_cm2.isna().sum()),
        'high_performance_threshold_uA_mM_cm2':THRESHOLD,
        'threshold_origin':'Frozen upper-quartile criterion from the governed model-input set at initial classification definition.',
        'high_performance_rows':int(df.HighPerformanceLabel.sum()),
        'chemistry_counts':{str(k):int(v) for k,v in Counter(df.BroadChemistry).items()},
        'source_audit_status_counts':{str(k):int(v) for k,v in Counter(df.SourceAuditStatus).items()} if 'SourceAuditStatus' in df.columns else {},
        'important_boundary':'All 151 evidence configurations are retained for model learning; inverse-domain restriction occurs only at the candidate decision stage.'
    }
    (TAB/'database_QA_v5.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa,indent=2))

if __name__=='__main__': main()
