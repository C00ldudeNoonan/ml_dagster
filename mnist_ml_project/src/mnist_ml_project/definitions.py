import dagster as dg
from pathlib import Path
from mnist_ml_project.defs.resources import LocalModelStoreResource, ComputeResource

# Create resource instances
local_model_storage = LocalModelStoreResource(models_path="./models")
compute_config = ComputeResource(device="cpu", batch_size=32, max_workers=4)

# Dynamically load all definitions from the defs folder and merge with resources
defs = dg.Definitions.merge(
    dg.load_from_defs_folder(project_root=Path(__file__).parent.parent),
    dg.Definitions(
        resources={
            "model_storage": local_model_storage,
            "compute": compute_config
        }
    )
)
