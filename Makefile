THIS_DIR=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))


.PHONY: build-antlr-image
build-antlr-image:
	cd $(PWD)/bootstrap/grammar/ && docker build -t serene-antlr:latest .

.PHONY: gen-parser-go
gen-parser-go:
	docker run -it --rm --user $(shell id -u):$(shell id -g) -v $(PWD):/serene serene-antlr:latest -Dlanguage=Go -o /serene/bootstrap/pkg/parser/ /serene/bootstrap/grammar/Serene.g4

.PHONY: lint
lint:
	cd $(THIS_DIR)/bootstrap && cargo fmt -- --check

.PHONY: test-bootstrap
test-bootstrap:
	cd $(THIS_DIR)/bootstrap && cargo test

.PHONY: test
test: test-bootstrap


.PHONY: clean-bootstrap
clean-bootstrap:
	cd $(THIS_DIR)/bootstrap && cargo clean

.PHONY: bootstrap-repl
bootstrap-repl:
	cd $(THIS_DIR)/bootstrap && cargo run repl

.PHONY: clean
clean: clean-bootstrap
