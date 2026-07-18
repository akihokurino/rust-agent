-include Makefile.local

.PHONY: smoke
smoke:
	cargo run --example smoke