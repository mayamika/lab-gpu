CC := nvcc
CFLAGS := -Werror cross-execution-space-call -lm -D_MWAITXINTRIN_H_INCLUDED

ifneq ($(CPPSTD),)
CFLAGS += -std=$(CPPSTD)
endif

ifeq ($(BENCHMARK),true)
CFLAGS += -DBENCHMARK
endif

NBLOCKS ?= 256
NTHREADS ?= 256
CFLAGS += -DNBLOCKS=$(NBLOCKS) -DNTHREADS=$(NTHREADS)

TARGET := lab2

BINARIES_DIR := ./bin/
SOURCES_DIR := ./src/

all: $(TARGET)

$(TARGET): $(BINARIES_DIR)/$(TARGET).cu $(wildcard $(SOURCES_DIR)/*.cuh $(SOURCES_DIR)/*.cu)
	$(CC) $(CFLAGS) -o "$@" $< -I$(SOURCES_DIR)

.PHONY: clean
clean:
	$(RM) $(TARGET)
