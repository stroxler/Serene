ROOT_DIR=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))

include $(ROOT_DIR)/bootstrap/Makefile

.PHONY: compile
compile: compile-bootstrap

.PHONY: build
build: compile

clean: clean-bootstrap
