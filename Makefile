THIS_DIR=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: lint
lint:
	cd $(THIS_DIR)/bootstrap && cargo fmt -- --check

.PHONY: test-bootstrap
test-bootstrap:
	cd $(THIS_DIR)/bootstrap && cargo test

.PHONY: test
test: test-bootstrap
