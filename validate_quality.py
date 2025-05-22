import mlflow
import sys
from mlflow.tracking import MlflowClient

with open("val_accuracy.txt") as f:
    current_acc = float(f.read().strip())

client = MlflowClient()
experiment = client.get_experiment_by_name("mnist-experiment")

if not experiment:
    print("First run, no previous metrics")
    sys.exit(0)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=2
)

if len(runs) < 2:
    print("Only one run exists")
    sys.exit(0)

prev_acc = runs[1].data.metrics["val_accuracy"]

print(f"Current accuracy: {current_acc:.4f}, Previous: {prev_acc:.4f}")

if current_acc < prev_acc:
    print(f"Quality degraded! Failing pipeline. Current accuracy: {current_acc}, previous accuracy: {prev_acc}")
    sys.exit(1)
else:
    print("Quality maintained or improved")
