"""Print the README Results table from models/fairness_report.json."""

import json
from pathlib import Path

report = json.loads(
    (Path(__file__).resolve().parent.parent / "models" / "fairness_report.json").read_text()
)
overall = report["overall"]
sex = report["fairness"]["sex"]
race = report["fairness"]["race"]


def eo(attr):
    gaps, ci = attr["gaps"], attr["equalized_odds_ci"]
    return f"{gaps['equalized_odds_diff']:.3f} (95% CI {ci['low']:.3f} to {ci['high']:.3f})"


print("| Metric | Value |")
print("|---|---|")
print(f"| Accuracy | {overall['accuracy']:.3f} |")
print(f"| ROC AUC | {overall['roc_auc']:.3f} |")
print(f"| Equalized odds difference (sex) | {eo(sex)} |")
print(f"| Demographic parity difference (sex) | {sex['gaps']['demographic_parity_diff']:.3f} |")
print(f"| Equalized odds difference (race) | {eo(race)} |")
print(f"\nDataset: {report['dataset']} · test set: {report['n_test']} people")