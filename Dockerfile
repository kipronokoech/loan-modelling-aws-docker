# Use slim Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /opt/ml/code

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
# COPY src/ ./src/
COPY src/ /opt/ml/code/

# Set env variables for SageMaker compatibility
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Set default command (override in job spec)
# ENTRYPOINT ["python"]
CMD ["train.py"]
