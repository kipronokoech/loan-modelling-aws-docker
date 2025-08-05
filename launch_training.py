import boto3
import time

region = "us-east-1"
account_id = "993750298572"
bucket = "loan-modelling-69a3447a-2612-4dce-8144-cf053fa0db37"
role_arn = "arn:aws:iam::993750298572:role/SageMakerExecutionRole"

ecr_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/loan-model:v102"
# 993750298572.dkr.ecr.us-east-1.amazonaws.com/loan-model:v102
s3_input = f"s3://{bucket}/data/"
s3_output = f"s3://{bucket}/output/"
job_name = f"loan-model-training-{int(time.time())}"

sm = boto3.client("sagemaker", region_name=region)

response = sm.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        "TrainingImage": ecr_image,
        "TrainingInputMode": "File",
        "ContainerEntrypoint": ["python"],
        "ContainerArguments": ["train.py"]
        # "ContainerArguments": ["python", "-m", "loan_model.train"]
    },
    RoleArn=role_arn,
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_input,
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "text/csv",
            "InputMode": "File"
        }
    ],
    OutputDataConfig={
        "S3OutputPath": s3_output
    },
    ResourceConfig={
        "InstanceType": "ml.m5.large",
        "InstanceCount": 1,
        "VolumeSizeInGB": 10
    },
    StoppingCondition={
        "MaxRuntimeInSeconds": 1800
    },
    EnableManagedSpotTraining=False,
    CheckpointConfig={  # optional
        "S3Uri": f"s3://{bucket}/checkpoints/",
        "LocalPath": "/opt/ml/checkpoints"
    },
    RetryStrategy={"MaximumRetryAttempts": 1}
)

print(f"Training job started: {job_name}")
