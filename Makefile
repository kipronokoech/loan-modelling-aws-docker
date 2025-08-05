# Makefile

# Image name
IMAGE_NAME = loan-model

# Paths
TRAIN_INPUT_CSV = $(CURDIR)/data/train.csv
TEST_INPUT_CSV  = $(CURDIR)/data/test.csv
MODEL_DIR       = $(CURDIR)/models
OUTPUT_DIR      = $(CURDIR)/output

# Default target
.PHONY: all
all: build

# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

# Train the model locally
.PHONY: train-local
train-local:
	docker run --rm \
		-v "$(dir $(TRAIN_INPUT_CSV)):/opt/ml/input/data/train:ro" \
		-v "$(MODEL_DIR):/opt/ml/model" \
		$(IMAGE_NAME) \
		src/train.py

# Run inference locally
.PHONY: inference-local
inference-local:
	docker run --rm \
		-v "$(dir $(TEST_INPUT_CSV)):/opt/ml/input/data/test:ro" \
		-v "$(MODEL_DIR):/opt/ml/model" \
		-v "$(OUTPUT_DIR):/opt/ml/output" \
		$(IMAGE_NAME) \
		src/inference.py

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
