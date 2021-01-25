ROOT_DIR=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))

include $(ROOT_DIR)/bootstrap/Makefile

.PHONY: lint
lint: lint-bootstrap

.PHONY: compile
compile: compile-bootstrap

.PHONY: build
build: compile

clean: clean-bootstrap
