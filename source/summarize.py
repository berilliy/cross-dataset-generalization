"""Build article tables and static figures from one completed train run."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DOM = {"UNSW_NB15":"UNSW", "TON_IOT":"ToN", "CICIOT2023":"CIC"}
MODELS = ["Majority", "LogisticRegression", "RandomForest", "HistGradientBoosting", "LightGBM"]
SHORT = dict(zip(MODELS, ["Majority", "LR", "RF", "HGB", "LightGBM"]))
def label(s):
    return " + ".join(DOM.get(i, i) for i in s.split("+"))


def summarize(root):
    r = root / "run_results"
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    read = lambda name: pd.read_csv(r / (name+".csv"), float_precision="round_trip")
    def save(frame, name):
        frame.to_csv(tables / (name+".csv"), index=False)
    def figsave(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(figures / (name+"."+ext), dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "axes.labelcolor":"#273843","xtick.color":"#40525e","ytick.color":"#40525e",
                         "pdf.fonttype":42, "font.family":"DejaVu Sans"})
    b=read("controlled_benchmark")
    keys=["target","sources","scenario","seed","model","policy"]
    a=b.pivot(index=keys,columns="variant",values="macro_f1").reset_index()
    a["delta_duration"]=a.minimal_core-a.protocol_only
    assert len(a)==360 and a.notna().all().all()
    save(a,"duration_paired")
    ids=[k for k in keys if k!="seed"]
    paired=a.groupby(ids).agg(core=("minimal_core","mean"),protocol=("protocol_only","mean"),
                difference=("delta_duration","mean"),sd_difference=("delta_duration","std"),
                min_difference=("delta_duration","min"),max_difference=("delta_duration","max")).reset_index()
    save(paired,"duration_summary")
    keys=["target","sources","scenario","seed","variant","policy"]
    m=b.merge(b.loc[b.model.eq("Majority"),keys+["macro_f1"]].rename(columns={"macro_f1":"majority_macro_f1"}),
              on=keys,validate="many_to_one")
    m=m.loc[m.model.ne("Majority")].copy()
    m["delta_vs_majority"]=m.macro_f1-m.majority_macro_f1
    save(m[keys+["model","macro_f1","majority_macro_f1","delta_vs_majority"]],"majority_paired")
    mb=m.groupby([k for k in keys if k!="seed"]+["model"],as_index=False)[["macro_f1","majority_macro_f1","delta_vs_majority"]].mean()
    save(mb,"majority_summary")
    l=paired.query("model=='LightGBM' and policy=='default_0.5'").sort_values(["target","scenario","sources"])
    names=[label(row.sources)+" → "+label(row.target) for row in l.itertuples()]
    fig,ax=plt.subplots(figsize=(7.7,5.0),layout="constrained")
    y=np.arange(len(l))
    ax.hlines(y,l.protocol,l.core,color="#a0afb7",lw=2.2)
    ax.scatter(l.protocol,y,s=48,facecolor="white",edgecolor="#255775",linewidth=1.7,label="Protocol only",zorder=3)
    ax.scatter(l.core,y,s=38,color="#bf532d",label="Protocol + duration",zorder=4)
    ax.set(yticks=y,yticklabels=names,xlim=(0,1),xlabel="Macro-F1 · fixed threshold 0.5");ax.invert_yaxis()
    ax.grid(axis="x",alpha=.16);ax.legend(loc="upper center",bbox_to_anchor=(.5,1.12),ncol=2,frameon=False)
    figsave(fig,"02_duration")
    sel=mb.query("variant=='minimal_core' and policy=='default_0.5' and scenario!='in_domain'").copy()
    sel["direction"]=[label(s)+" → "+label(t) for s,t in zip(sel.sources,sel.target)]
    matrix=sel.pivot(index="direction",columns="model",values="delta_vs_majority")[MODELS[1:]]
    fig,ax=plt.subplots(figsize=(7.3,4.6),layout="constrained")
    lim=max(abs(matrix.to_numpy().ravel()))
    heat=ax.imshow(matrix,cmap="RdBu",vmin=-lim,vmax=lim,aspect="auto")
    ax.set(xticks=range(4),xticklabels=[SHORT[k] for k in matrix.columns],yticks=range(len(matrix)),yticklabels=matrix.index)
    for (i,j),v in np.ndenumerate(matrix.to_numpy()):
        ax.text(j,i,f"{v:+.3f}",ha="center",va="center",color="white" if abs(v)>.60*lim else "#17232a")
    fig.colorbar(heat,ax=ax,shrink=.82,label="ML minus Majority · Macro-F1")
    figsave(fig,"01_majority")
    assoc=read("shift_performance_association").query("variant=='minimal_core' and scenario!='in_domain' and model!='Majority'")
    cond=assoc.groupby(["target","sources","model","policy"],as_index=False)[["mean_ks","macro_f1_drop"]].mean()
    save(cond,"shift_conditions")
    rows=[]
    for (model,policy),z in cond.groupby(["model","policy"]):
        assert len(z)==9
        rows.append(dict(model=model,policy=policy,conditions=len(z),rho=spearmanr(z.mean_ks,z.macro_f1_drop).statistic))
    save(pd.DataFrame(rows),"shift_correlations")
    fig,axes=plt.subplots(1,4,figsize=(10.2,3.25),sharex=True,sharey=True,layout="constrained")
    palette={"CICIOT2023":"#ab4b36","UNSW_NB15":"#306b89","TON_IOT":"#358275"}
    for ax,model in zip(axes,MODELS[1:]):
        z=cond[(cond.model==model)&(cond.policy=="default_0.5")]
        for target,group in z.groupby("target"):
            ax.scatter(group.mean_ks,group.macro_f1_drop,color=palette[target],s=35,label=label(target))
        rho=spearmanr(z.mean_ks,z.macro_f1_drop).statistic
        ax.set_title(f"{SHORT[model]}\nρ = {rho:+.2f}");ax.grid(alpha=.18)
    axes[0].set_ylabel("In-domain minus transfer Macro-F1");fig.supxlabel("Mean marginal KS across five columns",fontsize=10)
    axes[-1].legend(loc="best",frameon=False,fontsize=8)
    figsave(fig,"03_shift")
    c=read("calibration_sensitivity")
    fig,axes=plt.subplots(2,2,figsize=(8.0,5.6),layout="constrained",sharex=True)
    for row,ds in enumerate(["UNSW_NB15","TON_IOT"]):
        for col,metric in enumerate(["fpr","recall"]):
            ax=axes[row,col]
            for window,color,marker in [("full","#183d59","o"),("early","#b14d2f","^"),("late","#267b70","s")]:
                z=c[(c.dataset==ds)&(c.model=="LightGBM")&(c.window==window)].sort_values("alpha",ascending=False)
                ax.plot(range(len(z)),z[metric]*100,color=color,marker=marker,label=window,alpha=.9)
            ax.set_title(label(ds)+" · "+("False alarms" if col==0 else "Attack recall"))
            ax.set(xticks=[0,1,2],xticklabels=["1%","0.5%","0.1%"]);ax.set_ylabel("Test rows in class (%)")
            ax.grid(alpha=.18)
            if metric=="recall":
                ax.set_ylim(-2,102)
                ax.set_yticks([0,25,50,75,100])
    axes[0,0].legend(frameon=False,fontsize=8);fig.supxlabel("Benign false-alarm budget on the calibration window",fontsize=10)
    figsave(fig,"04_calibration")
    e=read("external_metrics")
    cols=["macro_f1","fpr","recall","roc_auc","average_precision","feature_overlap","f1_at_20pct_attack"]
    es=e.groupby(["sources","variant","model","policy"])[cols].mean().reset_index()
    save(es,"external_summary")
    pooled="UNSW_NB15+TON_IOT"
    def pooling(frame, ids, name):
        pool=frame[frame.sources.str.contains(r"\+")][ids+["sources","macro_f1"]].rename(columns={"macro_f1":"pooled_macro_f1","sources":"pooled_sources"})
        single=frame[~frame.sources.str.contains(r"\+")].merge(pool,on=ids,validate="many_to_one")
        single["pooled_minus_single"]=single.pooled_macro_f1-single.macro_f1
        save(single[ids+["sources","pooled_sources","macro_f1","pooled_macro_f1","pooled_minus_single"]],name)
    pooling(b[b.scenario!="in_domain"],["target","seed","variant","model","policy"],"benchmark_pooling")
    pooling(e,["seed","variant","model","policy"],"external_pooling")
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout="constrained",sharey=True)
    for ax,variant in zip(axes,["minimal_core","protocol_only"]):
        for j,(s,clr) in enumerate([("UNSW_NB15","#286483"),("TON_IOT","#ca6c43"),(pooled,"#34887b")]):
            z=es[(es.variant==variant)&(es.policy=="default_0.5")&(es.sources==s)].set_index("model").loc[MODELS]
            ax.bar(np.arange(5)+(j-1)*.23,z.macro_f1,.22,label=label(s),color=clr)
        ax.set(xticks=np.arange(5),xticklabels=[SHORT[k] for k in MODELS],ylim=(0,1))
        ax.tick_params(axis="x",rotation=22);ax.grid(axis="y",alpha=.18)
        ax.set_title("Protocol + duration" if variant=="minimal_core" else "Protocol only")
    axes[0].set_ylabel("RT-IoT2022 · Macro-F1");axes[0].legend(frameon=False,fontsize=8)
    figsave(fig,"05_external")
    return es


if __name__=="__main__":
    summarize(Path(__file__).resolve().parents[1])

