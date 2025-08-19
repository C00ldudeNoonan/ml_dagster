import dagster as dg
from mnist_ml_project.defs.assets.data_assets import (
    raw_mnist_data,
    processed_mnist_data,
)
from mnist_ml_project.defs.assets.model_assets import (
    digit_classifier,
    model_evaluation,
    production_digit_classifier,
)
from mnist_ml_project.defs.assets.prediction_assets import (
    batch_digit_predictions,
    digit_predictions,
)
from mnist_ml_project.defs.jobs import (
    training_job,
    deployment_job,
    inference_job,
    full_pipeline_job,
)
from mnist_ml_project.defs.schedules import batch_inference_schedule
from mnist_ml_project.defs.sensors import model_failure_sensor
from mnist_ml_project.defs.resources import (
    LocalModelStoreResource,
    S3ModelStoreResource,
    ComputeResource,
)

# Define resource instances
local_model_storage = LocalModelStoreResource(models_path="./models")

compute_config = ComputeResource(device="cpu", batch_size=32, max_workers=4)

# Create the Definitions object
defs = dg.Definitions(
    assets=[
        raw_mnist_data,
        processed_mnist_data,
        digit_classifier,
        model_evaluation,
        production_digit_classifier,
        batch_digit_predictions,
        digit_predictions,
    ],
    jobs=[training_job, deployment_job, inference_job, full_pipeline_job],
    schedules=[batch_inference_schedule],
    sensors=[model_failure_sensor],
    resources={"model_storage": local_model_storage, "compute": compute_config},
)
