import dagster as dg


@dg.run_failure_sensor
def model_failure_sensor(context: dg.RunFailureSensorContext):
    """Alert when ML pipeline jobs fail."""

    if context.dagster_run.job_name in [
        "digit_classifier_training",
        "model_deployment",
    ]:
        return dg.SkipReason("Slack integration not configured")

    # In production, you would send alerts to Slack/email/PagerDuty
    # from dagster_slack import make_slack_on_run_failure_sensor
    #
    # model_failure_sensor = make_slack_on_run_failure_sensor(
    #     monitored_jobs=[training_job, deployment_job],
    #     slack_token=dg.EnvVar("SLACK_TOKEN"),
    #     channel="#ml-alerts",
    #     default_status="FAILURE",
    #     text_fn=lambda context: f"🚨 ML Pipeline Failure: {context.dagster_run.job_name}"
    # )
