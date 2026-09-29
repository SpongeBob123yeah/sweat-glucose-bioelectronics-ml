import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
RANDOM_STATE=42
THRESHOLD=3076.
BASE=Path(__file__).resolve().parents[1]
DB=BASE/'tables'/'final_database_151_v5.csv'
OUT=BASE/'results'; OUT.mkdir(parents=True,exist_ok=True)

def paper_weights(groups):
    c=Counter(groups); return np.array([1.0/c[g] for g in groups],float)

def make_prep(cat_features,n_features):
    return ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),list(range(len(cat_features)))),('num','passthrough',list(range(len(cat_features),n_features)))])

def wr2(y,p,w):
    yb=np.average(y,weights=w); ssr=np.sum(w*(y-p)**2); sst=np.sum(w*(y-yb)**2); return float(1-ssr/sst) if sst>0 else float('nan')

def metrics(y,p,w):
    return {'R2':wr2(y,p,w),'RMSE':float(np.sqrt(np.average((y-p)**2,weights=w))),'MAE':float(np.average(np.abs(y-p),weights=w)),'MRE_pct':float(np.average(np.abs(y-p)/np.maximum(np.abs(y),1e-9),weights=w)*100)}

def fit_reg(model,X,y,w,weighted=True):
    if weighted:
        try: model.fit(X,y,sample_weight=w); return model
        except TypeError: pass
    model.fit(X,y); return model

def reg_factory(name):
    if name=='XGB': return XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.10,min_child_weight=1,subsample=1.0,colsample_bytree=1.0,reg_lambda=1.0,reg_alpha=0.0,objective='reg:squarederror',random_state=RANDOM_STATE,n_jobs=1)
    if name=='ET': return ExtraTreesRegressor(n_estimators=160,max_depth=None,min_samples_leaf=1,random_state=RANDOM_STATE,n_jobs=1)
    if name=='RF': return RandomForestRegressor(n_estimators=160,max_depth=None,min_samples_leaf=1,random_state=RANDOM_STATE,n_jobs=1)
    if name=='SVR': return SVR(C=1.0,gamma='scale',epsilon=0.1,kernel='rbf')
    if name=='MLP': return MLPRegressor(hidden_layer_sizes=(40,),alpha=1e-4,learning_rate_init=1e-3,max_iter=1400,random_state=RANDOM_STATE)
    raise KeyError(name)

def cramers_v(x,y):
    tab=pd.crosstab(pd.Series(x),pd.Series(y)); n=tab.values.sum()
    if n==0 or min(tab.shape)<2: return 0.0
    chi2=chi2_contingency(tab,correction=False)[0]; phi2=chi2/n; r,k=tab.shape
    phi2corr=max(0,phi2-(k-1)*(r-1)/(n-1)); rcorr=r-(r-1)**2/(n-1); kcorr=k-(k-1)**2/(n-1); den=min(kcorr-1,rcorr-1)
    return float(np.sqrt(phi2corr/den)) if den>0 else 0.0

def eta(cat,cont):
    cats=pd.Series(cat).astype(str); y=np.asarray(cont,float); gm=np.mean(y); num=0
    for c in cats.unique():
        yy=y[cats.values==c]; num+=len(yy)*(np.mean(yy)-gm)**2
    den=np.sum((y-gm)**2); return float(np.sqrt(num/den)) if den>0 else 0.0

def main():
    FIG = BASE / 'figures'
    FIG.mkdir(exist_ok=True)
    df=pd.read_csv(DB,keep_default_na=False)
    df['HierarchicalFlag']=(df['Morphology']=='Nanoflower').astype(int)
    df['ChemMorph']=df['BroadChemistry']+'|'+df['Morphology']; df['ChemSupport']=df['BroadChemistry']+'|'+df['Support']; df['MorphSupport']=df['Morphology']+'|'+df['Support']
    df['LogSensitivity']=np.log10(1+df['Sensitivity_uA_mM_cm2'].astype(float)); df['HighPerformanceLabel']=(df['Sensitivity_uA_mM_cm2'].astype(float)>=THRESHOLD).astype(int)
    cat=['BroadChemistry','Morphology','Support','Modifier','SupportTopology','CollectorFamily','ChemMorph','ChemSupport','MorphSupport']; num=['IntegratedFlag','DirectGrowthFlag','BinderFreeFlag']; features=cat+num
    X=df[features].astype(object).to_numpy(); y=df['LogSensitivity'].to_numpy(float); yc=df['HighPerformanceLabel'].to_numpy(int); g=df['PaperGroupID'].astype(str).to_numpy()

    # Five-model regression benchmark: 5-fold paper-grouped CV; export true/pred records for Fig.2d
    outer = GroupKFold(n_splits=5)
    fold_rows = []
    pred_records = []
    model_names = ['XGB', 'ET', 'RF', 'SVR', 'MLP']

    for fold, (tr, te) in enumerate(outer.split(X, y, g), 1):
        prep = make_prep(cat, len(features))
        Xt = prep.fit_transform(X[tr])
        Xv = prep.transform(X[te])
        wtr = paper_weights(g[tr])
        wte = paper_weights(g[te])
        y_true_te = y[te]
        g_te = g[te]

        for name in model_names:
            m = fit_reg(reg_factory(name), Xt, y[tr], wtr, weighted=(name != 'MLP'))
            p = m.predict(Xv)
            fold_rows.append({'Fold': fold, 'Model': name, **metrics(y_true_te, p, wte)})
            for yi, pi, gi in zip(y_true_te, p, g_te):
                pred_records.append({
                    'Fold': fold,
                    'Model': name,
                    'PaperGroupID': gi,
                    'True_LogSensitivity': yi,
                    'Pred_LogSensitivity': pi
                })

    reg = pd.DataFrame(fold_rows)
    reg.to_csv(OUT / 'regression_fold_metrics.csv', index=False)
    reg_summary = reg.groupby('Model')[['R2', 'RMSE', 'MAE', 'MRE_pct']].agg(['mean', 'std']).reset_index()
    reg_summary.columns = ['_'.join([str(z) for z in c if z]).rstrip('_') for c in reg_summary.columns]
    reg_summary.to_csv(OUT / 'regression_summary.csv', index=False)

    pred_df = pd.DataFrame(pred_records)
    pred_df.to_csv(OUT / 'regression_pred_true_records.csv', index=False)
    print(f"Saved: {OUT / 'regression_fold_metrics.csv'}")
    print(f"Saved: {OUT / 'regression_pred_true_records.csv'}")

    # Nested grouped high-performance classifier. C is selected by a one-standard-error rule inside each outer training fold.
    c_grid=np.array([.005,.01,.02,.03,.05,.075,.1,.2,.5,1.0]); cout=[]; oof=np.zeros(len(df))
    sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
    for fold,(tr,te) in enumerate(sg.split(X,yc,g),1):
        prep=make_prep(cat,len(features)); Xt=prep.fit_transform(X[tr]); Xv=prep.transform(X[te])
        inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=RANDOM_STATE+fold); scores=[]
        for C in c_grid:
            ll=[]
            for a,b in inner.split(X[tr],yc[tr],g[tr]):
                pp=make_prep(cat,len(features)); Xa=pp.fit_transform(X[tr][a]); Xb=pp.transform(X[tr][b]); lr=LogisticRegression(C=float(C),class_weight='balanced',max_iter=5000,solver='liblinear',random_state=RANDOM_STATE); lr.fit(Xa,yc[tr][a],sample_weight=paper_weights(g[tr][a])); pr=lr.predict_proba(Xb)[:,1]; ll.append(log_loss(yc[tr][b],pr,sample_weight=paper_weights(g[tr][b]),labels=[0,1]))
            scores.append((C,float(np.mean(ll)),float(np.std(ll,ddof=1)/np.sqrt(len(ll)))))
        best=min(scores,key=lambda z:z[1]); cutoff=best[1]+best[2]; Csel=min([z for z in scores if z[1]<=cutoff],key=lambda z:z[0])[0]
        lr=LogisticRegression(C=float(Csel),class_weight='balanced',max_iter=5000,solver='liblinear',random_state=RANDOM_STATE); lr.fit(Xt,yc[tr],sample_weight=paper_weights(g[tr])); pr=lr.predict_proba(Xv)[:,1]; oof[te]=pr; w=paper_weights(g[te]); cout.append({'Fold':fold,'Selected_C':float(Csel),'LogLoss':float(log_loss(yc[te],pr,sample_weight=w,labels=[0,1])),'ROC_AUC':float(roc_auc_score(yc[te],pr,sample_weight=w)),'AP':float(average_precision_score(yc[te],pr,sample_weight=w)),'Brier':float(np.average((yc[te]-pr)**2,weights=w))})
    cdf=pd.DataFrame(cout); cdf.to_csv(OUT/'classifier_fold_metrics.csv',index=False); pd.DataFrame({'RecordID':df.RecordID,'PaperGroupID':df.PaperGroupID,'Observed':yc,'OOF_Probability':oof}).to_csv(OUT/'classifier_oof_predictions.csv',index=False)

    # Final regularization on all literature data, selected with the same grouped one-SE rule.
    sg2=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE); fullscores=[]
    for C in c_grid:
        ll=[]
        for a,b in sg2.split(X,yc,g):
            pp=make_prep(cat,len(features)); Xa=pp.fit_transform(X[a]); Xb=pp.transform(X[b]); lr=LogisticRegression(C=float(C),class_weight='balanced',max_iter=5000,solver='liblinear',random_state=RANDOM_STATE); lr.fit(Xa,yc[a],sample_weight=paper_weights(g[a])); pr=lr.predict_proba(Xb)[:,1]; ll.append(log_loss(yc[b],pr,sample_weight=paper_weights(g[b]),labels=[0,1]))
        fullscores.append((float(C),float(np.mean(ll)),float(np.std(ll,ddof=1)/np.sqrt(len(ll)))))
    best=min(fullscores,key=lambda z:z[1]); cutoff=best[1]+best[2]; final_C=min([z for z in fullscores if z[1]<=cutoff],key=lambda z:z[0])[0]
    pd.DataFrame(fullscores,columns=['C','MeanLogLoss','SELogLoss']).to_csv(OUT/'classifier_regularization_selection.csv',index=False)
    prep=make_prep(cat,len(features)); Xe=prep.fit_transform(X); lr=LogisticRegression(C=final_C,class_weight='balanced',max_iter=5000,solver='liblinear',random_state=RANDOM_STATE); lr.fit(Xe,yc,sample_weight=paper_weights(g))

    # SHAP interpretation of the final linear probabilistic classifier.
    expl = None
    try:
        import shap
        expl = shap.LinearExplainer(lr, Xe)
        sv = np.asarray(expl.shap_values(Xe))
    except Exception:
        sv = np.asarray(Xe.multiply(lr.coef_).toarray() if hasattr(Xe, 'multiply') else Xe * lr.coef_)

    names_out = list(prep.get_feature_names_out());
    imp = defaultdict(float)
    for j, n in enumerate(names_out):
        if n.startswith('cat__x0_'):
            grp = 'Active chemistry'
        elif n.startswith('cat__x1_'):
            grp = 'Morphology'
        elif n.startswith('cat__x2_') or n.startswith('cat__x4_') or n == 'num__x9':
            grp = 'Support/integration'
        elif n.startswith('cat__x3_'):
            grp = 'Modifier'
        elif n.startswith('cat__x5_'):
            grp = 'Current collector'
        elif n.startswith('cat__x6_') or n.startswith('cat__x7_') or n.startswith('cat__x8_'):
            grp = 'Architecture interactions'
        elif n == 'num__x10':
            grp = 'Direct growth'
        elif n == 'num__x11':
            grp = 'Binder-free'
        else:
            grp = 'Other'
        imp[grp] += float(np.mean(np.abs(sv[:, j])))
    pd.DataFrame(sorted(imp.items(), key=lambda z: -z[1]), columns=['DescriptorGroup', 'MeanAbsSHAP']).to_csv(
        OUT / 'shap_grouped.csv', index=False)

    if expl is not None:
        group_df = pd.read_csv(OUT / 'shap_grouped.csv')
        group_names = group_df['DescriptorGroup'].tolist()

        feat_to_group = {}
        for j, n in enumerate(names_out):
            if n.startswith('cat__x0_'):
                grp = 'Active chemistry'
            elif n.startswith('cat__x1_'):
                grp = 'Morphology'
            elif n.startswith('cat__x2_') or n.startswith('cat__x4_') or n == 'num__x9':
                grp = 'Support/integration'
            elif n.startswith('cat__x3_'):
                grp = 'Modifier'
            elif n.startswith('cat__x5_'):
                grp = 'Current collector'
            elif n.startswith('cat__x6_') or n.startswith('cat__x7_') or n.startswith('cat__x8_'):
                grp = 'Architecture interactions'
            elif n == 'num__x10':
                grp = 'Direct growth'
            elif n == 'num__x11':
                grp = 'Binder-free'
            else:
                grp = 'Other'
            feat_to_group[j] = grp

        n_sample = sv.shape[0]
        agg_shap = np.zeros((n_sample, len(group_names)))
        if hasattr(Xe, "toarray"):
            X_arr = Xe.toarray()
        else:
            X_arr = Xe
        for g_idx, gname in enumerate(group_names):
            cols = [j for j, g in feat_to_group.items() if g == gname]
            agg_shap[:, g_idx] = np.sum(sv[:, cols], axis=1)

        shap_agg_exp = shap.Explanation(values=agg_shap, feature_names=group_names)
        plt.figure(figsize=(10, 6))
        fixed_order = list(range(len(group_names)))

        shap.plots.beeswarm(shap_agg_exp, show=False, order=fixed_order, color="#1883ff", color_bar=False)
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        start = int(np.ceil(x_min * 10))
        stop = int(np.floor(x_max * 10))
        new_ticks = np.arange(start, stop + 1, 1) / 10.0
        ax.set_xticks(new_ticks)
        ax.tick_params(axis='x', labelsize=8)

        plt.tight_layout()
        plt.savefig(FIG / "shap_beeswarm_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()


    else:
        print("WARNING: LinearExplainer failed, skip beeswarm plot")

    assoc_cols = ['BroadChemistry', 'Morphology', 'Support', 'SupportTopology', 'CollectorFamily', 'Modifier',
                  'IntegratedFlag', 'DirectGrowthFlag', 'HierarchicalFlag', 'LogSensitivity'];
    types = {c: 'cat' for c in assoc_cols};
    types['LogSensitivity'] = 'cont';
    A = np.eye(len(assoc_cols))
    for i, a in enumerate(assoc_cols):
        for j, b in enumerate(assoc_cols):
            if j <= i: continue
            if types[a] == 'cont' and types[b] == 'cont':
                v = abs(spearmanr(df[a], df[b]).statistic)
            elif types[a] == 'cat' and types[b] == 'cat':
                v = cramers_v(df[a].astype(str), df[b].astype(str))
            elif types[a] == 'cat':
                v = eta(df[a].astype(str), df[b].astype(float))
            else:
                v = eta(df[b].astype(str), df[a].astype(float))
            A[i, j] = A[j, i] = v
    pd.DataFrame(A, index=assoc_cols, columns=assoc_cols).to_csv(OUT / 'association_matrix.csv')

    phases=['CuOH2-containing','NiOH2-containing','CoOHx-containing']
    morphs=['0D','1D','2D','Nanoflower']
    supports=[('Graphene','Graphene3D','GrapheneIntegrated'),('CNT','CNTIntegrated','CNTIntegrated')]
    cand=[]
    for ph in phases:
        for mo in morphs:
            for sup,topo,col in supports:
                r={'BroadChemistry':ph,'Morphology':mo,'Support':sup,'Modifier':'None',
                   'SupportTopology':topo,'CollectorFamily':col,'IntegratedFlag':1,
                   'DirectGrowthFlag':1,'BinderFreeFlag':1,
                   'HierarchicalFlag':1 if mo=='Nanoflower' else 0}
                r['ChemMorph']=ph+'|'+mo; r['ChemSupport']=ph+'|'+sup; r['MorphSupport']=mo+'|'+sup
                cand.append(r)
    C=np.array([[r[f] for f in features] for r in cand],object)
    phigh=lr.predict_proba(prep.transform(C))[:,1]
    hier=np.array([r['HierarchicalFlag'] for r in cand],float)

    # Weight-free Pareto dominance: maximize P_high and H_hierarchy.
    def dominance_count(i, active=None):
        ids=range(len(cand)) if active is None else active
        n=0
        for j in ids:
            if j==i: continue
            weak=(phigh[j]>=phigh[i]) and (hier[j]>=hier[i])
            strict=(phigh[j]>phigh[i]) or (hier[j]>hier[i])
            if weak and strict: n+=1
        return n

    # Full non-dominated sorting into Pareto fronts.
    remaining=set(range(len(cand))); front=np.zeros(len(cand),dtype=int); fno=1
    while remaining:
        this=[]
        for i in sorted(remaining):
            dominated=False
            for j in remaining:
                if j==i: continue
                weak=(phigh[j]>=phigh[i]) and (hier[j]>=hier[i])
                strict=(phigh[j]>phigh[i]) or (hier[j]>hier[i])
                if weak and strict:
                    dominated=True; break
            if not dominated: this.append(i)
        if not this: raise RuntimeError('Pareto sorting failed')
        for i in this: front[i]=fno
        remaining.difference_update(this); fno+=1

    # Scalar decision utility is applied only AFTER weight-free Pareto analysis.
    wh=.10
    merit=(1-wh)*phigh+wh*hier
    decision_order=np.argsort(-merit)
    p_order=np.argsort(-phigh)
    p_rank={idx:pos for pos,idx in enumerate(p_order,1)}
    decision_rank={idx:pos for pos,idx in enumerate(decision_order,1)}
    target_i=next(i for i,r in enumerate(cand) if r['BroadChemistry']=='CuOH2-containing' and r['Morphology']=='Nanoflower' and r['Support']=='Graphene')

    # Literature support counts are provenance descriptors, not ranking terms.
    rr=[]
    for idx,r in enumerate(cand):
        core=((df.BroadChemistry==r['BroadChemistry'])&(df.Morphology==r['Morphology'])&(df.Support==r['Support'])).sum()
        full=((df.BroadChemistry==r['BroadChemistry'])&(df.Morphology==r['Morphology'])&(df.Support==r['Support'])&
              (df.Modifier=='None')&(df.IntegratedFlag.astype(int)==1)&(df.DirectGrowthFlag.astype(int)==1)&(df.BinderFreeFlag.astype(int)==1)).sum()
        rr.append({
            'DecisionRank':int(decision_rank[idx]),
            'PHighRank':int(p_rank[idx]),
            'ParetoFront':int(front[idx]),
            'ParetoNonDominated':bool(front[idx]==1),
            'DominationCount':int(dominance_count(idx)),
            'Chemistry':r['BroadChemistry'],
            'Morphology':r['Morphology'],
            'Support':r['Support'],
            'P_high':float(phigh[idx]),
            'H_hierarchy':float(hier[idx]),
            'PerformanceWeight':float(1-wh),
            'HierarchyWeight':float(wh),
            'DecisionUtility':float(merit[idx]),
            'ObservedCoreTripleCount':int(core),
            'ObservedFullArchitectureCount':int(full),
            'Target':bool(idx==target_i)
        })
    rdfull=pd.DataFrame(rr).sort_values('DecisionRank')
    rdfull.to_csv(OUT/'candidate_decision_table_24_v5.csv',index=False)
    rdfull.to_csv(BASE/'tables'/'candidate_decision_v5.csv',index=False)
    rdfull[rdfull.ParetoNonDominated].sort_values('P_high',ascending=False).to_csv(OUT/'pareto_front_v5.csv',index=False)

    # Weight sensitivity of the final target and the full decision order.
    rob=[]
    allr=[]
    for wh2 in np.arange(0, 51, 1) / 100.0:
        sc=(1-wh2)*phigh+wh2*hier; od=np.argsort(-sc)
        ranks={idx:pos for pos,idx in enumerate(od,1)}
        rob.append({'HierarchyWeight':float(round(wh2,2)),
                    'TargetRank':int(ranks[target_i]),
                    'TargetDecisionUtility':float(sc[target_i])})
        for idx in range(len(cand)):
            allr.append({'HierarchyWeight':float(round(wh2,2)),'CandidateIndex':idx,
                         'Chemistry':cand[idx]['BroadChemistry'],'Morphology':cand[idx]['Morphology'],
                         'Support':cand[idx]['Support'],'Rank':int(ranks[idx]),'DecisionUtility':float(sc[idx])})
    robdf=pd.DataFrame(rob); robdf.to_csv(OUT/'rank_robustness_v5.csv',index=False)
    pd.DataFrame(allr).to_csv(OUT/'all_candidate_weight_sensitivity_v5.csv',index=False)
    minw=float(robdf.loc[robdf.TargetRank==1,'HierarchyWeight'].min())

    # Pareto summary and final JSON.
    pf=rdfull[rdfull.ParetoNonDominated].sort_values(['H_hierarchy','P_high'],ascending=[False,False])
    summary={
        'version':'v5.0 PARETO-LOCKED',
        'dataset':{'n_configurations':int(len(df)),'n_paper_groups':int(df.PaperGroupID.nunique()),
                   'threshold_uA_mM_cm2':THRESHOLD,
                   'threshold_rationale':'Frozen upper-quartile criterion from the governed model-input set at initial classification definition.'},
        'regression_target':'log10(1+sensitivity)',
        'regression_summary':reg.groupby('Model')[['R2','RMSE','MAE','MRE_pct']].mean().reset_index().to_dict(orient='records'),
        'classifier_nested_cv_mean':cdf[['LogLoss','ROC_AUC','AP','Brier']].mean().to_dict(),
        'classifier_final_C':float(final_C),
        'classifier_C_selection_rule':'strongest regularization within one standard error of minimum grouped-CV log-loss',
        'inverse_scope':'modifier-minimized single-metal transition-metal hydroxides integrated with nanoscale carbon supports (graphene or CNT)',
        'candidate_count':24,
        'pareto_objectives':['maximize P_high','maximize H_hierarchy'],
        'pareto_front_size':int(len(pf)),
        'pareto_front':pf[['Chemistry','Morphology','Support','P_high','H_hierarchy','DecisionUtility']].to_dict(orient='records'),
        'decision_utility':'M = 0.90 * P_high + 0.10 * H_hierarchy; applied only after Pareto analysis',
        'hierarchy_weight':float(wh),
        'min_weight_for_target_rank1':minw,
        'target_decision_rank':int(rdfull.loc[rdfull.Target,'DecisionRank'].iloc[0]),
        'target_p_high_rank':int(rdfull.loc[rdfull.Target,'PHighRank'].iloc[0]),
        'target_pareto_front':int(rdfull.loc[rdfull.Target,'ParetoFront'].iloc[0]),
        'top5_decision':rdfull.head(5).to_dict(orient='records'),
        'claim_boundary':'Rank 1 is implementation-domain decision ranking; the target is also weight-free Pareto-non-dominated. No global-optimum claim.'
    }
    (OUT/'final_results_v5.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
