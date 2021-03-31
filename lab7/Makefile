CC := mpic++
CFLAGS := -fopenmp -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -lm

ifneq ($(CPPSTD),)
CFLAGS += -std=$(CPPSTD)
endif

ifeq ($(BENCHMARK),true)
CFLAGS += -DBENCHMARK
endif

# DEBUG ?= true
ifeq ($(DEBUG),true)
CFLAGS += -DDEBUG
endif

NBLOCKS ?= 256
NTHREADS ?= 512
CFLAGS += -DNBLOCKS=$(NBLOCKS) -DNTHREADS=$(NTHREADS)

TARGET := lab7

BINARIES_DIR := ./bin/
SOURCES_DIR := ./src/

all: $(TARGET)

$(TARGET): $(BINARIES_DIR)/$(TARGET).cpp $(wildcard $(SOURCES_DIR)/*.cuh $(SOURCES_DIR)/*.cu)
	$(CC) $(CFLAGS) -o "$@" $< -I$(SOURCES_DIR)

.PHONY: clean
clean:
	$(RM) $(TARGET)
