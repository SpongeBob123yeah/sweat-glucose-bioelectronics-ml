from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE=Path(__file__).resolve().parents[1]
FIG=BASE/'figures'; RES=BASE/'results'; TAB=BASE/'tables'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'figure.dpi':150})

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.tif',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def shortlab(r):
    ch=str(r.Chemistry).replace('-containing','').replace('CuOH2','Cu(OH)2').replace('NiOH2','Ni(OH)2')
    mo='NF' if r.Morphology=='Nanoflower' else str(r.Morphology)
    sup='G' if r.Support=='Graphene' else 'CNT'
    return f'{ch} / {mo} / {sup}'

df=pd.read_csv(TAB/'final_database_151_v5.csv',keep_default_na=False)
reg=pd.read_csv(RES/'regression_fold_metrics.csv')
cfold=pd.read_csv(RES/'classifier_fold_metrics.csv')
shap=pd.read_csv(RES/'shap_grouped.csv')
assoc=pd.read_csv(RES/'association_matrix.csv',index_col=0)
rank=pd.read_csv(RES/'candidate_decision_table_24_v5.csv')
pareto=pd.read_csv(RES/'pareto_front_v5.csv')
robust=pd.read_csv(RES/'rank_robustness_v5.csv')
summary=json.loads((RES/'final_results_v5.json').read_text())

# --------------------------- Main Fig. 2 ---------------------------
fig=plt.figure(figsize=(15.3,10.2),constrained_layout=True)
gs=fig.add_gridspec(3,4,height_ratios=[1.0,1.02,1.22])
axs={
'a':fig.add_subplot(gs[0,0]),'b':fig.add_subplot(gs[0,1]),'c':fig.add_subplot(gs[0,2:4]),
'd':fig.add_subplot(gs[1,0]),'e':fig.add_subplot(gs[1,1]),'f':fig.add_subplot(gs[1,2]),'g':fig.add_subplot(gs[1,3]),
'h':fig.add_subplot(gs[2,:])}

# a: broad evidence chemistry distribution
ax=axs['a']; vc=df.BroadChemistry.value_counts().sort_values(); y=np.arange(len(vc))
pretty={'CuOxide-containing':'Cu oxide','CuOH2-containing':'Cu(OH)2','NiOH2-containing':'Ni(OH)2','CoOHx-containing':'CoOHx','Other':'Other'}
ax.barh(y,vc.values); ax.set_yticks(y,[pretty.get(x,x) for x in vc.index]); ax.set_xlabel('Configurations'); ax.set_title('Broad literature evidence space')
for yy,v in zip(y,vc.values): ax.text(v+0.6,yy,str(v),va='center',fontsize=8)

# b: mixed association matrix
ax=axs['b']; im=ax.imshow(assoc.values,aspect='auto',vmin=0,vmax=1)
labs=['Chemistry','Morphology','Support','Support topology','Collector','Modifier','Integrated','Direct growth','Hierarchical','log sensitivity']
xlabs=['Chem','Morph','Support','Topology','Collector','Modifier','Integrated','Direct','Hier.','log S']
ax.set_xticks(range(len(labs)),xlabs,rotation=50,ha='right'); ax.set_yticks(range(len(labs)),labs); ax.set_title('Mixed-type association strength')
cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.03); cb.set_label('Association (0–1)')

# c: workflow with explicit Pareto step
ax=axs['c']; ax.axis('off'); ax.set_title('Evidence-to-decision workflow')
boxes=[
(.01,.49,.14,.27,'151-config\nevidence space'),
(.18,.49,.14,.27,'Grouped\nregression'),
(.35,.49,.14,.27,'Nested grouped\nclassifier'),
(.52,.49,.14,.27,'24 candidates\nall scored'),
(.69,.49,.14,.27,'Weight-free\nPareto front'),
(.86,.49,.13,.27,'Final utility\nranking')]
for x0,y0,w,h,t in boxes:
    p=FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.012'); ax.add_patch(p); ax.text(x0+w/2,y0+h/2,t,ha='center',va='center',fontsize=7.4)
for i in range(len(boxes)-1):
    x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]
    ax.annotate('',xy=(x2,.625),xytext=(x1,.625),arrowprops={'arrowstyle':'->','shrinkA':2,'shrinkB':2})
ax.text(.01,.17,'No morphology is prefiltered. Pareto dominance is evaluated on all 24 candidates using P_high and the independent hierarchical-architecture objective before the 90/10 decision utility is applied.',fontsize=7.8,ha='left',va='center',wrap=True)

# d: regression benchmark
ax=axs['d']; fam=['XGB','ET','RF','SVR','MLP']; x=np.arange(5); means=[]; sds=[]
for n in fam:
    z=reg[reg.Model==n].R2; means.append(z.mean()); sds.append(z.std(ddof=1))
ax.errorbar(x,means,yerr=sds,fmt='o',capsize=3); ax.axhline(0,linewidth=.8); ax.set_xticks(x,fam,rotation=25,ha='right'); ax.set_ylabel('Grouped-CV $R^2$'); ax.set_title('Cross-paper regression diagnostic')

# e: classifier validation
ax=axs['e']; metnames=['ROC_AUC','AP','LogLoss','Brier']; disp=['ROC-AUC','AP','Log-loss','Brier']; vals=[cfold[m].mean() for m in metnames]; errs=[cfold[m].std(ddof=1) for m in metnames]; xx=np.arange(4)
ax.bar(xx,vals,yerr=errs,capsize=3); ax.set_xticks(xx,disp,rotation=30,ha='right'); ax.set_ylim(0,1); ax.set_ylabel('Metric value'); ax.set_title('Nested grouped classifier')
for i,v in enumerate(vals): ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=7.5)

# f: grouped SHAP
ax=axs['f']; sh=shap.sort_values('MeanAbsSHAP'); yy=np.arange(len(sh)); ax.barh(yy,sh.MeanAbsSHAP); ax.set_yticks(yy,sh.DescriptorGroup); ax.set_xlabel('Mean |SHAP value|'); ax.set_title('Architecture-level model importance')

# g: weight-free Pareto objective space
ax=axs['g']
non=rank[~rank.ParetoNonDominated.astype(bool)]; pf=rank[rank.ParetoNonDominated.astype(bool)]
ax.scatter(non.P_high,non.H_hierarchy,s=26,alpha=.65,label='Dominated')
ax.scatter(pf.P_high,pf.H_hierarchy,s=85,marker='*',label='Pareto front',zorder=4)
ax.set_yticks([0,1],['Non-hierarchical','Hierarchical']); ax.set_ylim(-.22,1.22); ax.set_xlim(rank.P_high.min()-.008,rank.P_high.max()+.009)
ax.set_xlabel('$P_{high}$'); ax.set_title('Weight-free Pareto analysis')
for _,r in pf.iterrows():
    lab='Cu(OH)$_2$/NF/G' if r.Morphology=='Nanoflower' else 'Cu(OH)$_2$/1D/G'
    dy=.08 if r.H_hierarchy==1 else .08
    ax.annotate(lab,(r.P_high,r.H_hierarchy),xytext=(4,7),textcoords='offset points',fontsize=7)
ax.legend(frameon=False,loc='center left',bbox_to_anchor=(.02,.50))

# h: decision ranking after Pareto reporting
ax=axs['h']; top=rank.sort_values('DecisionRank').head(12).iloc[::-1].copy(); yy=np.arange(len(top))
labels=[]
for _,r in top.iterrows(): labels.append(shortlab(r)+('  ★' if bool(r.ParetoNonDominated) else ''))
ax.barh(yy,top.DecisionUtility,label='Decision utility $M$')
ax.scatter(top.P_high,yy,marker='o',label='$P_{high}$',zorder=3)
ax.set_yticks(yy,labels); ax.set_xlim(0.40,0.60); ax.set_xlabel('Score / propensity'); ax.set_title('Final performance-dominant decision ranking after Pareto analysis')
ax.legend(frameon=False,ncol=2,loc='upper right')
for y0,r in enumerate(top.itertuples()): ax.text(r.DecisionUtility+.002,y0,f'#{r.DecisionRank}',va='center',fontsize=7)
ax.text(.01,-.12,'★ weight-free Pareto-non-dominated candidate;  M = 0.90 P_high + 0.10 H_hierarchy',transform=ax.transAxes,fontsize=7.7,va='top',clip_on=False)

for k,ax in axs.items(): ax.text(-.12,1.07,k,transform=ax.transAxes,fontweight='bold',fontsize=12,va='top')
save(fig,'Fig2_analysis_source_panels')

# --------------------------- Supplementary S1 ---------------------------
fig,axs=plt.subplots(1,2,figsize=(9.2,4.1),constrained_layout=True)
order=['Other','3D','Nanoflower','2D','1D','0D']; vc=df.Morphology.value_counts(); vals=[vc.get(k,0) for k in order]; y=np.arange(len(order))
axs[0].barh(y,vals); axs[0].set_yticks(y,order); axs[0].set_xlabel('Count'); axs[0].set_title('Active-material morphology')
for yy,v in zip(y,vals): axs[0].text(v+.5,yy,str(v),va='center',fontsize=8)
sv=df.Support.replace({'None':'No functional support'}).value_counts().sort_values(); y=np.arange(len(sv)); axs[1].barh(y,sv.values); axs[1].set_yticks(y,sv.index); axs[1].set_xlabel('Count'); axs[1].set_title('Functional support')
for yy,v in zip(y,sv.values): axs[1].text(v+.5,yy,str(v),va='center',fontsize=8)
axs[0].text(-.12,1.05,'a',transform=axs[0].transAxes,fontweight='bold',fontsize=12); axs[1].text(-.12,1.05,'b',transform=axs[1].transAxes,fontweight='bold',fontsize=12)
save(fig,'Supplementary_Figure_S1_v5')

# S2-S5 regression metrics
pairs=[('R2','Grouped-CV $R^2$','Supplementary_Figure_S2_v5'),('RMSE','RMSE (log target)','Supplementary_Figure_S3_v5'),('MRE_pct','MRE (%)','Supplementary_Figure_S4_v5'),('MAE','MAE (log target)','Supplementary_Figure_S5_v5')]
fam=['XGB','ET','RF','SVR','MLP']
for metric,ylabel,name in pairs:
    fig,ax=plt.subplots(1,1,figsize=(5.6,4.2),constrained_layout=True)
    data=[reg[reg.Model==n][metric].to_numpy() for n in fam]; ax.boxplot(data,showmeans=True); ax.set_xticks(range(1,6),fam); ax.set_ylabel(ylabel); ax.set_title('Paper-grouped five-fold regression benchmark')
    if metric=='R2': ax.axhline(0,linewidth=.8)
    for j,z in enumerate(data,1): ax.scatter(np.full(len(z),j),z,s=16,zorder=3)
    save(fig,name)

# Table S1 descriptor schema. Hierarchical objective is explicitly NOT a classifier predictor in v5.
rows=[
('BroadChemistry','Categorical','CuOH2-containing; NiOH2-containing; CoOHx-containing; CuOxide-containing; Other','One-hot','Broad active-chemistry family'),
('Morphology','Categorical','0D; 1D; 2D; 3D; Nanoflower; Other','One-hot','Reported active-material morphology'),
('Support','Categorical','Graphene; CNT; CarbonFabric; ConductivePolymer; Other; None','One-hot','Functional support; separated from current collector'),
('Modifier','Categorical','None plus literature-reported modifier classes','One-hot','Separates promoter/composite effects from the base active phase'),
('SupportTopology','Categorical','Graphene3D; graphene coating; carbon fabric; metal-self-supported; etc.','One-hot','Architecture topology'),
('CollectorFamily','Categorical','GrapheneIntegrated; CNTIntegrated; GCE; ITO/FTO; metal foils/foams; etc.','One-hot','Current-collector family'),
('IntegratedFlag','Binary','0/1','Binary predictor','Structurally integrated architecture'),
('DirectGrowthFlag','Binary','0/1','Binary predictor','Direct growth of active phase on support/collector'),
('BinderFreeFlag','Binary','0/1','Binary predictor','Binder-free architecture'),
('ChemMorph','Interaction','Chemistry × morphology','One-hot interaction','Coupled chemistry–morphology effect'),
('ChemSupport','Interaction','Chemistry × support','One-hot interaction','Coupled chemistry–support effect'),
('MorphSupport','Interaction','Morphology × support','One-hot interaction','Coupled morphology–support effect'),
('H_hierarchy','Decision objective','1 for nanoflower; 0 for 0D/1D/2D','Not used as an independent classifier predictor','Independent second objective for Pareto/decision analysis; chemistry- and support-independent')]
pd.DataFrame(rows,columns=['Descriptor','Type','Values_or_categories','Encoding','Role']).to_csv(TAB/'descriptor_schema_v5.csv',index=False)
print('v5 figures and supplementary tables generated')
