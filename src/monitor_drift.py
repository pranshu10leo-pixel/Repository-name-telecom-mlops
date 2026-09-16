import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE = Path("data/telecom_master.csv")
OUTPUT = Path("drift_report.html")

# Load reference data
reference_data = pd.read_csv(REFERENCE)

# Create a simulated current production batch
current_data = reference_data.sample(
    frac=0.4,
    random_state=42
).copy()

# Introduce deliberate distribution changes
current_data["revenue_inr"] = current_data["revenue_inr"] * 1.35
current_data["data_used_gb"] = current_data["data_used_gb"] * 0.60

# Remove target and ID from monitoring
drop_columns = ["customer_id", "churn"]

reference_monitoring = reference_data.drop(columns=drop_columns)
current_monitoring = current_data.drop(columns=drop_columns)

# Generate drift report
report = Report([
    DataDriftPreset()
])

result = report.run(
    reference_data=reference_monitoring,
    current_data=current_monitoring
)

result.save_html(str(OUTPUT))

print(f"Drift report saved to: {OUTPUT}")
print("Simulated drift introduced in: revenue_inr, data_used_gb")
