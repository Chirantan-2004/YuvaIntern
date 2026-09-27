"""
Week 2 - Advanced Data Visualization & Storytelling
Titanic passenger dataset (cleaned in Week 1)

Run: python generate_visualizations.py
Input : titanic_cleaned.csv
Output: viz1_sankey_flow.png ... viz6_bubble_age_fare_family.png

Requires: pandas, matplotlib, seaborn, plotly, kaleido==0.2.1
(kaleido 0.2.1 is used deliberately -- newer kaleido versions require a
separate Chrome install for static image export)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("titanic_cleaned.csv")
df["SexLabel"] = df["Sex"].map({"male": "Male", "female": "Female"})
df["ClassLabel"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
df["SurvLabel"] = df["Survived"].map({0: "Did Not Survive", 1: "Survived"})

# =========================================================
# Figure 1 - Sankey diagram: Class -> Sex -> Survival Outcome
# =========================================================
class_order = ["1st Class", "2nd Class", "3rd Class"]
sex_order = ["Female", "Male"]
surv_order = ["Survived", "Did Not Survive"]
labels = class_order + sex_order + surv_order
label_idx = {l: i for i, l in enumerate(labels)}

flow1 = df.groupby(["ClassLabel", "SexLabel"], observed=True).size().reset_index(name="count")
flow2 = df.groupby(["SexLabel", "SurvLabel"], observed=True).size().reset_index(name="count")

sources, targets, values = [], [], []
for _, r in flow1.iterrows():
    sources.append(label_idx[r["ClassLabel"]])
    targets.append(label_idx[r["SexLabel"]])
    values.append(r["count"])
for _, r in flow2.iterrows():
    sources.append(label_idx[r["SexLabel"]])
    targets.append(label_idx[r["SurvLabel"]])
    values.append(r["count"])

node_colors = ["#2c3e50", "#34495e", "#7f8c8d", "#e91e8c", "#3498db", "#27ae60", "#c0392b"]
link_colors = []
for s, t in zip(sources, targets):
    if labels[t] == "Survived":
        link_colors.append("rgba(39,174,96,0.5)")
    elif labels[t] == "Did Not Survive":
        link_colors.append("rgba(192,57,41,0.5)")
    else:
        link_colors.append("rgba(160,160,160,0.35)")

fig = go.Figure(data=[go.Sankey(
    node=dict(pad=22, thickness=24, line=dict(color="white", width=1), label=labels, color=node_colors),
    link=dict(source=sources, target=targets, value=values, color=link_colors)
)])
fig.update_layout(title_text="Passenger Flow: Class -> Sex -> Survival Outcome",
                   font=dict(size=16, family="Arial"), width=1150, height=650,
                   margin=dict(t=70, l=20, r=20, b=20))
fig.write_image("viz1_sankey_flow.png", scale=2)
print("Saved viz1_sankey_flow.png")

# =========================================================
# Figure 2 - Split violin plot: Age distribution by survival and sex
# =========================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.violinplot(data=df, x="SurvLabel", y="Age", hue="SexLabel", split=True, ax=ax,
                palette={"Male": "#3498db", "Female": "#e91e8c"}, inner="quartile")
ax.set_title("Age Distribution by Survival Outcome, Split by Sex", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Age (years)")
ax.legend(title="Sex", loc="upper right")
plt.tight_layout()
plt.savefig("viz2_violin_age_survival.png")
plt.close()
print("Saved viz2_violin_age_survival.png")

# =========================================================
# Figure 3 - Heatmap: survival rate by age band x class
# =========================================================
bins = [0, 12, 18, 30, 45, 60, 100]
band_labels = ["0-12\n(Child)", "13-18\n(Teen)", "19-30\n(Young Adult)",
               "31-45\n(Adult)", "46-60\n(Older Adult)", "60+\n(Senior)"]
df["AgeBand"] = pd.cut(df["Age"], bins=bins, labels=band_labels, right=True)

pivot = df.groupby(["AgeBand", "ClassLabel"], observed=True)["Survived"].mean().unstack() * 100
pivot = pivot[["1st Class", "2nd Class", "3rd Class"]]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=50, ax=ax,
            cbar_kws={"label": "Survival Rate (%)"}, linewidths=1, linecolor="white",
            vmin=0, vmax=100)
ax.set_title("Survival Rate (%) by Age Group and Passenger Class", fontsize=14, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Age Group")
plt.tight_layout()
plt.savefig("viz3_heatmap_age_class_survival.png")
plt.close()
print("Saved viz3_heatmap_age_class_survival.png")

# =========================================================
# Figure 4 - 100% stacked bar: survival by embarkation port
# =========================================================
port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
df["PortLabel"] = df["Embarked"].map(port_map)
ct = pd.crosstab(df["PortLabel"], df["Survived"], normalize="index") * 100
ct.columns = ["Did Not Survive", "Survived"]
ct = ct.loc[["Southampton", "Cherbourg", "Queenstown"]]

fig, ax = plt.subplots(figsize=(8, 5))
ct.plot(kind="barh", stacked=True, ax=ax, color=["#c0392b", "#27ae60"], width=0.6)
ax.set_xlabel("Share of Passengers (%)")
ax.set_ylabel("")
ax.set_title("Survival Outcome by Port of Embarkation", fontsize=14, fontweight="bold")
for i, port in enumerate(ct.index):
    surv_pct = ct.loc[port, "Survived"]
    ax.text(101, i, f"{surv_pct:.0f}% survived", va="center", fontsize=10)
ax.set_xlim(0, 130)
ax.legend(loc="lower right", title=None)
plt.tight_layout()
plt.savefig("viz4_stacked_bar_embarkation.png")
plt.close()
print("Saved viz4_stacked_bar_embarkation.png")

# =========================================================
# Figure 5 - Treemap: passenger composition by class, sex, outcome
# =========================================================
grp = df.groupby(["ClassLabel", "SexLabel", "SurvLabel"], observed=True).size().reset_index(name="Count")

fig = px.treemap(
    grp, path=["ClassLabel", "SexLabel", "SurvLabel"], values="Count",
    color="SurvLabel",
    color_discrete_map={"Survived": "#27ae60", "Did Not Survive": "#c0392b", "(?)": "#cccccc"},
    title="Passenger Composition: Class -> Sex -> Survival Outcome (sized by passenger count)"
)
fig.update_traces(textinfo="label+value+percent parent", textfont_size=14)
fig.update_layout(font=dict(size=15, family="Arial"), margin=dict(t=60, l=10, r=10, b=10),
                   width=1100, height=650)
fig.write_image("viz5_treemap_composition.png", scale=2)
print("Saved viz5_treemap_composition.png")

# =========================================================
# Figure 6 - Faceted bubble chart: age, fare, family size, class, survival
# =========================================================
fig = px.scatter(
    df, x="Age", y="Fare", size="FamilySize", color="SurvLabel",
    facet_col="ClassLabel", facet_col_spacing=0.06,
    category_orders={"ClassLabel": ["1st Class", "2nd Class", "3rd Class"],
                      "SurvLabel": ["Survived", "Did Not Survive"]},
    color_discrete_map={"Survived": "#27ae60", "Did Not Survive": "#c0392b"},
    size_max=22, opacity=0.6,
    title="Age, Fare & Family Size by Class and Survival Outcome (bubble size = family size)"
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(font=dict(size=14, family="Arial"), width=1250, height=550,
                   legend_title_text="Outcome", margin=dict(t=80, l=60, r=20, b=60))
fig.write_image("viz6_bubble_age_fare_family.png", scale=2)
print("Saved viz6_bubble_age_fare_family.png")

print("\nAll 6 visualizations generated.")
