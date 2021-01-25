# compilation
CC = nvcc
CFLAGS = -Werror cross-execution-space-call -lm -std=c++11

# target
TARGET = lab1

# paths
BINARIES_DIR = ./bin/
SOURCES_DIR = ./src/

all: $(TARGET)

$(TARGET): $(BINARIES_DIR)/$(TARGET).cu $(SOURCES_DIR)/*.cuh
	$(CC) $(CFLAGS) -o $(TARGET) $(BINARIES_DIR)/$(TARGET).cu -I$(SOURCES_DIR)

.PHONY: clean
clean:
	$(RM) $(TARGET)
