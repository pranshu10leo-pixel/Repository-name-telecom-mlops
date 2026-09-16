import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE = Path("data/telecom_master.csv")
CURRENT = Path("data/telecom_master.csv")
OUTPUT = Path("drift_report.html")

reference_data = pd.read_csv(REFERENCE)
current_data = pd.read_csv(CURRENT)

report = Report([
    DataDriftPreset()
])

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

result.save_html(str(OUTPUT))

print(f"Drift report saved to: {OUTPUT}")
