# compilation
CC := nvcc
CFLAGS := -Werror cross-execution-space-call -lm -D_MWAITXINTRIN_H_INCLUDED

CPPSTD ?=
ifneq ($(CPPSTD),)
CFLAGS += -std=$(CPPSTD)
endif

NBLOCKS ?= 256
NTHREADS ?= 256
CFLAGS += -DNBLOCKS=$(NBLOCKS) -DNTHREADS=$(NTHREADS)

ifeq ($(BENCHMARK),true)
CFLAGS += -DBENCHMARK
endif

# target
TARGET := lab4

# paths
BINARIES_DIR := ./bin/
SOURCES_DIR := ./src/

all: $(TARGET)

$(TARGET): $(BINARIES_DIR)/$(TARGET).cu $(wildcard $(SOURCES_DIR)/*.cuh $(SOURCES_DIR)/*.cu)
	$(CC) $(CFLAGS) -o "$@" $(BINARIES_DIR)/$(TARGET).cu -I$(SOURCES_DIR)

.PHONY: clean
clean:
	$(RM) $(TARGET)
