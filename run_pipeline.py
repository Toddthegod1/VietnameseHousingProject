import subprocess
import sys

files = [
    # Original pipeline
    "src/preprocess_timeseries.py",
    "src/train_timeseries.py",
    "src/evaluate_timeseries.py",
    "src/cluster_districts.py",
    # New analyses
    "src/merge_external_data.py",
    "src/multi_horizon_forecast.py",
    "src/advanced_analysis.py",
]

for file in files:
    print(f"\nRunning {file}...")
    subprocess.run([sys.executable, file], check=True)

print("\nPipeline complete.")
