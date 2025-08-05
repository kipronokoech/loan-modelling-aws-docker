# Makefile

# Image name
IMAGE_NAME = loan-model
TAG := v102
IMAGE := $(IMAGE_NAME):$(TAG)

# Paths
TRAIN_INPUT_CSV = $(CURDIR)/data/train.csv
TEST_INPUT_CSV  = $(CURDIR)/data/test.csv
MODEL_DIR       = $(CURDIR)/models
OUTPUT_DIR      = $(CURDIR)/output
SRC_DIR = $(CURDIR)/src

# Default target
.PHONY: all
all: build

# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE) . --platform linux/amd64 

# Train the model locally
.PHONY: train-local
train-local:
	docker run --rm \
		-v "$(dir $(TRAIN_INPUT_CSV)):/opt/ml/input/data/train:ro" \
		-v "$(MODEL_DIR):/opt/ml/model" \
		-v "$(SRC_DIR):/opt/ml/code" \
		-w /opt/ml/code \
		--platform linux/amd64 \
		$(IMAGE) \
		python train.py

.PHONY: inference-local
inference-local:
	docker run --rm \
		-v "$(abspath data):/opt/ml/input/data/test:ro" \
		-v "$(abspath models):/opt/ml/model" \
		-v "$(abspath output):/opt/ml/output" \
		-v "$(SRC_DIR):/opt/ml/code" \
		-w /opt/ml/code \
		--platform linux/amd64 \
		$(IMAGE) \
		python inference.py \
			--input /opt/ml/input/data/test/test.csv \
			--model /opt/ml/model/loan_model.joblib \
			--features /opt/ml/model/feature_columns.joblib \
			--output /opt/ml/output


# Clean all artifacts
.PHONY: clean
clean:
	rm -rf $(MODEL_DIR)/*.joblib $(OUTPUT_DIR)/*.csv $(OUTPUT_DIR)/*.json

.PHONY: prune
prune:
	docker system prune -af

# Rebuild everything
.PHONY: rebuild
rebuild: clean build train-local inference-local


# REPO := loan-model
# IMAGE_AWS := $(REPO):$(TAG)

AWS_ACCOUNT_ID := 993750298572
AWS_REGION := us-east-1
ECR_REPO := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(IMAGE)

# build-aws:
# 	docker build . -t $(IMAGE) --platform linux/amd64 

push:
	docker tag $(IMAGE) $(ECR_REPO)
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	aws ecr describe-repositories --repository-names $(IMAGE_NAME) || aws ecr create-repository --repository-name $(IMAGE_NAME)
	docker push $(ECR_REPO)