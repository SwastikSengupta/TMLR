import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.3,"figure.dpi":200,"savefig.bbox":"tight"})
RED,BLUE,GREY="#c0392b","#2471a3","#7f8c8d"
c1=json.load(open("results/class1_estimation.json"))
ca=json.load(open("results/capacity_analysis.json"))
D=json.load(open("results/class3_per_dataset.json")); PER,META=D["per_dataset"],D["meta"]

# Fig 1: Class I replication
fig,ax=plt.subplots(figsize=(3.8,2.6))
d=np.array([r["dAUC"] for r in c1["rows"]])
ax.hist(d,bins=25,color=BLUE,alpha=.8)
ax.axvline(0.005,ls="--",c=RED,lw=1.2); ax.axvline(-0.005,ls="--",c=RED,lw=1.2)
ax.set_xlabel(r"$\Delta$AUC (leaky $-$ correct)"); ax.set_ylabel("conditions")
ax.set_title(f"Class I: {len(d)} conditions, all within $\\pm$0.005",fontsize=9)
plt.savefig("figures/fig1_class1.png"); plt.close()

# Fig 2: capacity predicts leakage
fig,ax=plt.subplots(figsize=(4.2,3.0))
ms=ca["models"]; x=[ca["measured_capacity"][m] for m in ms]; y=[ca["leakage"][m] for m in ms]
col=[RED if META[m]["locality"]=="local" else BLUE for m in ms]
ax.scatter(x,y,c=col,s=45,zorder=3)
for m,xi,yi in zip(ms,x,y):
    ax.annotate(m.replace("_","\n"),(xi,yi),fontsize=5.5,xytext=(3,3),textcoords="offset points")
z=np.polyfit(x,y,1); xs=np.linspace(min(x),max(x),50)
ax.plot(xs,np.polyval(z,xs),c=GREY,lw=1,ls="--",zorder=1)
ax.set_xlabel("measured capacity (random-label train AUC)")
ax.set_ylabel(r"Class III leakage ($\Delta$AUC)")
ax.set_title(f"Spearman $\\rho$ = {ca['spearman_rho']:.3f}",fontsize=9)
plt.savefig("figures/fig2_capacity.png"); plt.close()

# Fig 3: per-model effects, sorted
fig,ax=plt.subplots(figsize=(4.6,2.8))
order=sorted(ms,key=lambda m: ca["leakage"][m])
vals=[ca["leakage"][m] for m in order]
errs=[np.std([PER[m][d] for d in PER[m]],ddof=1)/np.sqrt(len(PER[m])) for m in order]
cols=[RED if META[m]["locality"]=="local" else BLUE for m in order]
ax.barh(range(len(order)),vals,xerr=errs,color=cols,alpha=.85)
ax.set_yticks(range(len(order))); ax.set_yticklabels([m.replace("_"," ") for m in order],fontsize=7)
ax.set_xlabel(r"Class III leakage ($\Delta$AUC), 15 datasets, $\pm$SE")
ax.set_title("red = tree/instance based, blue = pooled fit",fontsize=8)
plt.savefig("figures/fig3_models.png"); plt.close()
print("3 figures written")
