-include Makefile.local

.PHONY: test
test:
	cargo test --lib smoke_invoke -- --ignored --nocapture