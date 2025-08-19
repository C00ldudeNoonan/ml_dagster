import dagster as dg
from mnist_ml_project.defs.jobs import inference_job


batch_inference_schedule = dg.ScheduleDefinition(
    name="batch_inference_schedule",
    job=inference_job,
    cron_schedule="0 2 * * *",
    description="Process overnight batch of user-uploaded digit images",
)
