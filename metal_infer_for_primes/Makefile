# Makefile for metal_infer — Pure C/Metal MoE inference engine
#
# Targets:
#   make          — build metal_infer binary
#   make run      — single expert forward pass
#   make verify   — verify Metal vs CPU reference
#   make bench    — benchmark single expert (10 iterations)
#   make moe      — full MoE forward pass (K experts, single layer)
#   make moebench — benchmark MoE (10 iterations)
#   make full     — full 60-layer MoE forward pass (K=4)
#   make fullbench — benchmark full 60-layer forward (3 iterations)
#   make chat     — build interactive chat TUI
#   make clean    — remove build artifacts
#
# Note: Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

TARGET = metal_infer
EXTRA_C = hdgl_bootloaderz.c hdgl_router.c
MAIN_SRC = main.m

# Optional: offline shader compilation (faster startup, but not required)
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = shaders.metal
SHADER_AIR = shaders.air
SHADER_LIB = shaders.metallib

# Inference engine (complete forward pass)
INFER_TARGET = infer
INFER_SRC = infer.m

# Non-Metal CPU backend scaffold
NONMETAL_TARGET = nonmetal_infer
NONMETAL_SRC = nonmetal_infer.c hdgl_bootloaderz.c hdgl_router.c

# Chat TUI (interactive multi-turn)
CHAT_TARGET = chat
CHAT_SRC = chat.m

.PHONY: all clean run verify bench moe moebench full fullbench fast metallib infer infer-run chat-run build-chat nonmetal-check nonmetal-run nonmetal-moe nonmetal-embed nonmetal-lmhead nonmetal-route nonmetal-route-hdgl nonmetal-route-stack nonmetal-route-stack-hdgl

all: $(TARGET) $(INFER_TARGET) $(NONMETAL_TARGET)

# Build the binary (shaders compiled at runtime from source)
$(TARGET): $(MAIN_SRC) $(SHADER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(MAIN_SRC) -o $(TARGET)

# Optional: pre-compile shaders (not required — runtime compilation is the default)
metallib: $(SHADER_LIB)

$(SHADER_AIR): $(SHADER_SRC)
	$(METALC) -c $(SHADER_SRC) -o $(SHADER_AIR)

$(SHADER_LIB): $(SHADER_AIR)
	$(METALLIB_TOOL) $(SHADER_AIR) -o $(SHADER_LIB)

# Build the inference engine
$(INFER_TARGET): $(INFER_SRC) $(EXTRA_C)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(INFER_SRC) $(EXTRA_C) -o $(INFER_TARGET)

# Build the non-Metal backend scaffold
$(NONMETAL_TARGET): $(NONMETAL_SRC)
	$(CC) -O2 -Wall -Wextra $(NONMETAL_SRC) -o $(NONMETAL_TARGET) -lm

# Build the chat client (thin HTTP/SSE client + linenoise line editor)
$(CHAT_TARGET): $(CHAT_SRC) linenoise.c linenoise.h $(EXTRA_C)
	$(CC) -O2 -Wall -fobjc-arc -framework Foundation $(CHAT_SRC) linenoise.c $(EXTRA_C) -o $(CHAT_TARGET)

generate_hdgl_lattice: hdgl_lattice_generator.c $(EXTRA_C)
	$(CC) -O3 -Wall -I. hdgl_lattice_generator.c $(EXTRA_C) -o generate_hdgl_lattice -lm

# Pre-seed a lattice file (run once before inference for faster startup)
# Usage: make hdgl-preseed  ->  creates hdgl_lattice.bin
hdgl-preseed: generate_hdgl_lattice
	./generate_hdgl_lattice --manifest model_weights.json --steps 200 --output hdgl_lattice.bin

clean:
	rm -f $(TARGET) $(INFER_TARGET) $(NONMETAL_TARGET) $(CHAT_TARGET) $(SHADER_AIR) $(SHADER_LIB) generate_hdgl_lattice hdgl_lattice.bin

# Run targets
run: $(TARGET)
	./$(TARGET) --layer 0 --expert 0

verify: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --verify

fast: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --verify

bench: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --benchmark

moe: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe

moebench: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe --benchmark

full: $(TARGET)
	./$(TARGET) --fast --full --k 4

fullbench: $(TARGET)
	./$(TARGET) --fast --full --k 4 --benchmark

# Inference engine targets
build-infer: $(INFER_TARGET)

infer-run: $(INFER_TARGET)
	./$(INFER_TARGET) --prompt "Hello, what is" --tokens 20 --k 4

# Non-Metal backend targets
nonmetal-check: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --check-only

nonmetal-run: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --layer 0 --expert 0

nonmetal-moe: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --layer 0 --expert 0 --moe --k 4

nonmetal-embed: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --embed-token 0

nonmetal-lmhead: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --lm-head-token 0

nonmetal-route: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --route-token 0 --route-layer 0 --k 4

nonmetal-route-hdgl: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --route-token 0 --route-layer 0 --k 4 --hdgl

nonmetal-route-stack: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --route-token 0 --route-layer 0 --route-layers 4 --route-lm-head --k 4

nonmetal-route-stack-hdgl: $(NONMETAL_TARGET)
	./$(NONMETAL_TARGET) --route-token 0 --route-layer 0 --route-layers 4 --route-lm-head --k 4 --hdgl

# Chat TUI targets (use: make chat)

chat-run: $(CHAT_TARGET)
	./$(CHAT_TARGET) --k 4
