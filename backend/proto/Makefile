.PHONY: all clean python rust typescript install-deps

# Directories
PROTO_SRC_DIR := defs
PYTHON_OUT_DIR := gen/python
RUST_OUT_DIR := ../libs/rust/proto/src
TYPESCRIPT_OUT_DIR := ../../frontend/packages/protobuf/src

# Proto files
PROTO_FILES := $(wildcard $(PROTO_SRC_DIR)/*.proto)

# Default target
all: python rust typescript

# Install required dependencies
install-deps:
	@echo "📦 Installing protobuf dependencies..."
	@pip install protobuf grpcio-tools mypy-protobuf
	@npm install -g @protobuf-ts/plugin
	@cargo install protobuf-codegen

# Python codegen
python: $(PYTHON_OUT_DIR)
	@echo "🐍 Generating Python protobuf code..."
	@mkdir -p $(PYTHON_OUT_DIR)
	@python -m grpc_tools.protoc \
		--proto_path=$(PROTO_SRC_DIR) \
		--python_out=$(PYTHON_OUT_DIR) \
		--grpc_python_out=$(PYTHON_OUT_DIR) \
		--mypy_out=$(PYTHON_OUT_DIR) \
		$(PROTO_FILES)
	@echo "# Generated Python protobuf files" > $(PYTHON_OUT_DIR)/__init__.py
	@echo "✅ Python protobuf generation complete"

# Rust codegen
rust: $(RUST_OUT_DIR)
	@echo "🦀 Generating Rust protobuf code..."
	@mkdir -p $(RUST_OUT_DIR)
	@protoc \
		--proto_path=$(PROTO_SRC_DIR) \
		--rust_out=$(RUST_OUT_DIR) \
		$(PROTO_FILES)
	@echo "// Generated Rust protobuf files" > $(RUST_OUT_DIR)/mod.rs
	@echo "pub mod realtime;" >> $(RUST_OUT_DIR)/mod.rs
	@echo "pub mod control;" >> $(RUST_OUT_DIR)/mod.rs
	@echo "pub mod jobs;" >> $(RUST_OUT_DIR)/mod.rs
	@echo "✅ Rust protobuf generation complete"

# TypeScript codegen 
typescript: $(TYPESCRIPT_OUT_DIR)
	@echo "📜 Generating TypeScript protobuf code..."
	@mkdir -p $(TYPESCRIPT_OUT_DIR)
	@protoc \
		--proto_path=$(PROTO_SRC_DIR) \
		--ts_out=$(TYPESCRIPT_OUT_DIR) \
		--ts_opt=optimize_speed \
		--ts_opt=long_type_string \
		--ts_opt=generate_dependencies \
		$(PROTO_FILES)
	@echo "// Generated TypeScript protobuf files" > $(TYPESCRIPT_OUT_DIR)/index.ts
	@echo "export * from './realtime';" >> $(TYPESCRIPT_OUT_DIR)/index.ts
	@echo "export * from './control';" >> $(TYPESCRIPT_OUT_DIR)/index.ts
	@echo "export * from './jobs';" >> $(TYPESCRIPT_OUT_DIR)/index.ts
	@echo "✅ TypeScript protobuf generation complete"

# Create output directories
$(PYTHON_OUT_DIR):
	@mkdir -p $(PYTHON_OUT_DIR)

$(RUST_OUT_DIR):
	@mkdir -p $(RUST_OUT_DIR)

$(TYPESCRIPT_OUT_DIR):
	@mkdir -p $(TYPESCRIPT_OUT_DIR)

# Clean generated files
clean:
	@echo "🧹 Cleaning generated protobuf files..."
	@rm -rf $(PYTHON_OUT_DIR)/*
	@rm -rf $(RUST_OUT_DIR)/*
	@rm -rf $(TYPESCRIPT_OUT_DIR)/*
	@echo "✅ Clean complete"

# Development workflow
dev: clean all
	@echo "🚀 Development protobuf generation complete"

# Verify generation
verify:
	@echo "🔍 Verifying generated protobuf files..."
	@test -f $(PYTHON_OUT_DIR)/realtime_pb2.py || (echo "❌ Python generation failed"; exit 1)
	@test -f $(RUST_OUT_DIR)/realtime.rs || (echo "❌ Rust generation failed"; exit 1)
	@test -f $(TYPESCRIPT_OUT_DIR)/realtime.ts || (echo "❌ TypeScript generation failed"; exit 1)
	@echo "✅ All protobuf files generated successfully"

# Help
help:
	@echo "🔧 Available targets:"
	@echo "  all         - Generate all language bindings"
	@echo "  python      - Generate Python bindings"
	@echo "  rust        - Generate Rust bindings"
	@echo "  typescript  - Generate TypeScript bindings"
	@echo "  install-deps- Install required dependencies"
	@echo "  clean       - Remove generated files"
	@echo "  verify      - Verify generation completed"
	@echo "  help        - Show this help"