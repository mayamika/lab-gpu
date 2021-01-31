# compilation
CC := nvcc
CFLAGS := -Werror cross-execution-space-call -lm -D_MWAITXINTRIN_H_INCLUDED -std=c++11

# target
TARGET := lab2

# paths
BINARIES_DIR := ./bin/
SOURCES_DIR := ./src/

all: $(TARGET)

$(TARGET): $(BINARIES_DIR)/$(TARGET).cu $(wildcard $(SOURCES_DIR)/*.cuh $(SOURCES_DIR)/*.cu)
	$(CC) $(CFLAGS) -o "$@" $(BINARIES_DIR)/$(TARGET).cu -I$(SOURCES_DIR)

.PHONY: clean
clean:
	$(RM) $(TARGET)
