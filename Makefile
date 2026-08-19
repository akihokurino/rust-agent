-include Makefile.local

.PHONY: build fmt fmt-check lint test test-net check smoke

build:
	cargo build --all-targets
	cargo build --no-default-features

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --check

lint:
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --no-default-features -- -D warnings

test:
	cargo test

test-net:
	cargo test -- --ignored

check: fmt-check lint test

smoke:
	cargo run --example smoke