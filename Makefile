# Makefile

# Image name
IMAGE_NAME = loan-model
TAG := v102
IMAGE := $(IMAGE_NAME):$(TAG)

# Local paths
TRAIN_INPUT_CSV = $(CURDIR)/data/train.csv
TEST_INPUT_CSV  = $(CURDIR)/data/test.csv
MODEL_DIR       = $(CURDIR)/models
OUTPUT_DIR      = $(CURDIR)/output
SRC_DIR = $(CURDIR)/src

# Docker Mounts
CODE_DIR := /opt/ml/code
TRAIN_MOUNT := /opt/ml/input/data/train
TEST_MOUNT := /opt/ml/input/data/test
MODEL_MOUNT := /opt/ml/model
OUTPUT_MOUNT := /opt/ml/output

# File locations inside container
TRAIN_INPUT_FILE := $(TRAIN_MOUNT)/train.csv
TEST_INPUT_FILE := $(TEST_MOUNT)/test.csv
MODEL_FILE := $(MODEL_MOUNT)/loan_model.joblib
FEATURES_FILE := $(MODEL_MOUNT)/feature_columns.joblib
OUTPUT_FILE := $(OUTPUT_MOUNT)/predictions.csv

# Platform - to ensure compatibility with SageMaker
PLATFORM := linux/amd64

# Default target
.PHONY: all
all: build

# Build the Docker image
.PHONY: build
build:
	@docker build -t $(IMAGE) . --platform $(PLATFORM)

# Train the model locally
.PHONY: train-local
train-local:
	@docker run --rm \
		-v "$(dir $(TRAIN_INPUT_CSV)):$(TRAIN_MOUNT):ro" \
		-v "$(MODEL_DIR):$(MODEL_MOUNT)" \
		-v "$(SRC_DIR):$(CODE_DIR)" \
		--platform $(PLATFORM) \
		$(IMAGE) \
		python train.py

.PHONY: inference-local
inference-local:
	@docker run --rm \
		-v "$(abspath data):$(TEST_MOUNT):ro" \
		-v "$(abspath models):$(MODEL_MOUNT)" \
		-v "$(abspath output):$(OUTPUT_MOUNT)" \
		-v "$(SRC_DIR):$(CODE_DIR)" \
		--platform $(PLATFORM) \
		$(IMAGE) \
		python inference.py \
			--input $(TEST_INPUT_FILE) \
			--model $(MODEL_FILE) \
			--features $(FEATURES_FILE) \
			--output $(OUTPUT_FILE)

# Clean all artifacts
.PHONY: clean
clean:
	@rm -rf $(MODEL_DIR)/*.joblib $(OUTPUT_DIR)/*.csv $(OUTPUT_DIR)/*.json

.PHONY: prune
prune:
	@docker system prune -af

# Rebuild everything
.PHONY: rebuild
rebuild: prune clean build train-local inference-local

# Push AWS ECR
AWS_ACCOUNT_ID := 993750298572
AWS_REGION := us-east-1
ECR_REPO := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(IMAGE)

push:
	docker tag $(IMAGE) $(ECR_REPO)
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	aws ecr describe-repositories --repository-names $(IMAGE_NAME) || aws ecr create-repository --repository-name $(IMAGE_NAME)
	docker push $(ECR_REPO)