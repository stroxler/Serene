BOOTSTRAP_TAG ?= latest

compile-bootstrap:
	docker build -t serene-lang/serene-bootstrap:$(BOOTSTRAP_TAG) .

clean-bootstrap:
	docker rmi serene-lang/serene-bootstrap:$(BOOTSTRAP_TAG)
