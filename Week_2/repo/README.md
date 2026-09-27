# Week 2 — Advanced Data Visualization & Storytelling

Data science internship task: turn a cleaned dataset into a visual narrative a
non-technical audience can follow, using sophisticated chart types deliberately
matched to the story being told.

## Story
**"Women, Wealth & Survival: A Data Story of the Titanic"** — using the same cleaned
passenger dataset from Week 1, this project builds a sequence of 6 visualizations that
walk from a broad population overview down to a full multi-variable synthesis.

## Files
| File | Purpose |
|---|---|
| `titanic_cleaned.csv` | Cleaned dataset carried over from Week 1 |
| `generate_visualizations.py` | Generates all 6 charts used in the report |
| `viz1_sankey_flow.png` | Sankey diagram — passenger flow: class → sex → outcome |
| `viz2_violin_age_survival.png` | Split violin plot — age distribution by survival & sex |
| `viz3_heatmap_age_class_survival.png` | Heatmap — survival rate by age group × class |
| `viz4_stacked_bar_embarkation.png` | 100% stacked bar — survival by port of embarkation |
| `viz5_treemap_composition.png` | Treemap — passenger composition by class/sex/outcome |
| `viz6_bubble_age_fare_family.png` | Faceted bubble chart — age, fare, family size, class, outcome |
| `Week2_Titanic_Storytelling.docx` | Full write-up: narrative, chart rationale, code, implications |

## How to run
```bash
pip install pandas matplotlib seaborn plotly "kaleido==0.2.1"
python generate_visualizations.py
```
> Note: `kaleido==0.2.1` is used deliberately with `plotly==5.24.1` — newer kaleido
> versions require a separate Chrome install for static PNG export, which this
> pinned combination avoids.

## The narrative arc
1. **Sankey diagram** — who was on board, and where they ended up (opens the story)
2. **Violin plot** — did age protect you? (introduces age as a factor)
3. **Heatmap** — did age matter equally in every class? (shows the interaction)
4. **Stacked bar** — did embarkation port matter? (reframed as a class effect)
5. **Treemap** — human scale of each group (largest ≠ highest rate)
6. **Bubble chart** — age, fare, family size and outcome together (closing synthesis)

## Key insight
Survival was shaped far more by social position — sex, class, and the fare/location
that class implied — than by any individual trait like age alone. Children survived
more often, but that protection was far from equal across class (75% for 1st-class
children vs. 42% for 3rd-class children).

Full chart-by-chart rationale, code, and business/scientific implications are in
`Week2_Titanic_Storytelling.docx`.
