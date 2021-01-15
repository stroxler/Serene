THIS_DIR=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))

include $(THIS_DIR)/bootstrap/Makefile

.PHONY: compile
compile: compile-bootstrap

clean: clean-bootstrap
