BOOTSTRAP_TAG ?= latest
BOOTSTRAP_DIR = $(ROOT_DIR)/bootstrap

compile-bootstrap:
	docker build -t serene-lang/serene-bootstrap:$(BOOTSTRAP_TAG) $(BOOTSTRAP_DIR)

clean-bootstrap:
	docker rmi serene-lang/serene-bootstrap:$(BOOTSTRAP_TAG)
