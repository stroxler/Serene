BOOTSTRAP_TAG ?= latest
BOOTSTRAP_DIR = $(ROOT_DIR)/bootstrap

compile-bootstrap:
	docker build -t serene-lang/serene-bootstrap:$(BOOTSTRAP_TAG) $(BOOTSTRAP_DIR)

clean-bootstrap:
	docker rmi serene-lang/serene-bootstrap:$(BOOTSTRAP_TAG)

lint-bootstrap:
	docker run --rm -v $(BOOTSTRAP_DIR):/app -w /app golangci/golangci-lint:v1.35.2 golangci-lint run pkg/core
