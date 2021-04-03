CC := nvcc
CFLAGS := -ccbin=mpic++ -Werror cross-execution-space-call -lm -D_MWAITXINTRIN_H_INCLUDED

ifneq ($(CPPSTD),)
CFLAGS += -std=$(CPPSTD)
endif

ifeq ($(BENCHMARK),true)
CFLAGS += -DBENCHMARK
endif

ifeq ($(DEBUG),true)
CFLAGS += -DDEBUG
endif

NBLOCKS ?= 256
NTHREADS ?= 512
CFLAGS += -DNBLOCKS=$(NBLOCKS) -DNTHREADS=$(NTHREADS)

TARGET := lab8

BINARIES_DIR := ./bin/
SOURCES_DIR := ./src/

all: $(TARGET)

$(TARGET): $(BINARIES_DIR)/$(TARGET).cu $(wildcard $(SOURCES_DIR)/*.cuh $(SOURCES_DIR)/*.cu)
	$(CC) $(CFLAGS) -o "$@" $< -I$(SOURCES_DIR)

.PHONY: clean
clean:
	$(RM) $(TARGET)
