import boto3
from datetime import datetime

region = "us-east-1"
account_id = "993750298572"
bucket = "loan-modelling-69a3447a-2612-4dce-8144-cf053fa0db37"
role_arn = "arn:aws:iam::993750298572:role/SageMakerExecutionRole"
ecr_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/loan-model:v102"

model_name = "loan-model-inference"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
job_name = f"{model_name}-{timestamp}"

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

inference_processor = ScriptProcessor(
    image_uri=ecr_image,
    role=role_arn,
    instance_type="ml.t3.medium",
    instance_count=1,
    command=["python3"]
)

inference_processor.run(
    inputs=[
        ProcessingInput(source=f"s3://{bucket}/data/test.csv", destination="/opt/ml/processing/input"),
        ProcessingInput(source=f"s3://{bucket}/models/", destination="/opt/ml/processing/model")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{bucket}/inference/output/")
    ],
    code="src/inference.py",
    arguments=[
        "--input", "/opt/ml/processing/input/test.csv",
        "--model", "/opt/ml/processing/model/loan_model.joblib",
        "--features", "/opt/ml/processing/model/feature_columns.joblib",
        "--output", "/opt/ml/processing/output"
    ]

)
