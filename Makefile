.PHONY: bench bench-clean

FOLDER := target/release/examples
EXT :=
 # if windows, add .exe to the end of the file
ifeq ($(OS),Windows_NT)
	EXT := .exe
endif

# Binary names for each build
BIN_DEFAULT_NAME := tensor_operations_default
BIN_SIMD_NAME := tensor_operations_simd
BIN_PARALLEL_NAME := tensor_operations_parallel
BIN_SIMD_PARALLEL_NAME := tensor_operations_simd_parallel

BIN_DEFAULT := $(FOLDER)/tensor_operations_default
BIN_SIMD := $(FOLDER)/tensor_operations_simd
BIN_PARALLEL := $(FOLDER)/tensor_operations_parallel
BIN_SIMD_PARALLEL := $(FOLDER)/tensor_operations_simd_parallel

# Build all variants
build-default:
	@echo "Building with default features..."
	@cargo build --release --example tensor_operations --no-default-features
	@cp $(FOLDER)/tensor_operations$(EXT) $(BIN_DEFAULT)$(EXT) 2> /dev/null || cp $(FOLDER)/tensor_operations $(BIN_DEFAULT)

build-simd:
	@echo "Building with SIMD..."
	@cargo build --release --example tensor_operations --no-default-features --features simd
	@cp $(FOLDER)/tensor_operations$(EXT) $(BIN_SIMD)$(EXT) 2> /dev/null || cp $(FOLDER)/tensor_operations $(BIN_SIMD)

build-parallel:
	@echo "Building with Parallel..."
	@cargo build --release --example tensor_operations --no-default-features --features parallel
	@cp $(FOLDER)/tensor_operations$(EXT) $(BIN_PARALLEL)$(EXT) 2> /dev/null || cp $(FOLDER)/tensor_operations $(BIN_PARALLEL)

build-simd-parallel:
	@echo "Building with SIMD + Parallel..."
	@cargo build --release --example tensor_operations --no-default-features --features simd_and_parallel
	@cp $(FOLDER)/tensor_operations$(EXT) $(BIN_SIMD_PARALLEL)$(EXT) 2> /dev/null || cp $(FOLDER)/tensor_operations $(BIN_SIMD_PARALLEL)

# Clean build artifacts
bench-clean:
	@echo "Cleaning up..."
	@rm -f $(BIN_DEFAULT) $(BIN_SIMD) $(BIN_PARALLEL) $(BIN_SIMD_PARALLEL) \
	      $(BIN_DEFAULT)$(EXT) $(BIN_SIMD)$(EXT) $(BIN_PARALLEL)$(EXT) $(BIN_SIMD_PARALLEL)$(EXT) 2> /dev/null || true

# Run benchmarks
bench: bench-clean build-default build-simd build-parallel build-simd-parallel
	@echo "\n=== Running benchmarks (10 runs, 2 warmups) ===\n"
	@if command -v hyperfine >/dev/null 2>&1; then \
		echo "Hyperfine found, running benchmarks...$(BIN_DEFAULT)$(EXT) $(BIN_SIMD)$(EXT) $(BIN_PARALLEL)$(EXT) $(BIN_SIMD_PARALLEL)$(EXT)"; \
		cd $(FOLDER); \
		hyperfine \
			--warmup 2 \
			--runs 10 \
			--export-markdown BENCHMARKS.md \
			--setup 'echo "=== Running: {}"' \
			--show-output \
			--sort command \
			$(BIN_DEFAULT_NAME)$(EXT) \
			$(BIN_SIMD_NAME)$(EXT) \
			$(BIN_PARALLEL_NAME)$(EXT) \
			$(BIN_SIMD_PARALLEL_NAME)$(EXT); \
		cd -; \
		cp $(FOLDER)/BENCHMARKS.md .; \
		echo "\nBenchmark results saved to BENCHMARKS.md"; \
	else \
		echo "Hyperfine not found. Please install it with 'cargo install hyperfine' or from your package manager."; \
		echo "Would run the following commands:"; \
		echo "  $(BIN_DEFAULT)"; \
		echo "  $(BIN_SIMD)"; \
		echo "  $(BIN_PARALLEL)"; \
		echo "  $(BIN_SIMD_PARALLEL)"; \
	fi

# Alias for bench
benchmark: bench
