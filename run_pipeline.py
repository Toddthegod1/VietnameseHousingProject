import subprocess
import sys

files = [
    "src/preprocess_timeseries.py",
    "src/train_timeseries.py",
    "src/evaluate_timeseries.py",
    "src/cluster_districts.py",
]

for file in files:
    print(f"\nRunning {file}...")
    subprocess.run([sys.executable, file], check=True)

print("\nPipeline complete.")