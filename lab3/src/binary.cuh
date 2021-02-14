#ifndef BINARY_CUH
#define BINARY_CUH

#include <fstream>

namespace binary {
template <typename T>
void ReadBinary(std::ifstream& file, T& data) {
    file.read(static_cast<char*>(static_cast<void*>(&data)), sizeof(data));
}

template <typename T>
void WriteBinary(std::ofstream& file, const T& data) {
    file.write(static_cast<const char*>(static_cast<const void*>(&data)),
               sizeof(data));
}

template <typename T>
void ReadBinaryArray(std::ifstream& file, T* data, int size) {
    file.read(static_cast<char*>(static_cast<void*>(data)), size * sizeof(T));
}

template <typename T>
void WriteBinaryArray(std::ofstream& file, const T* data, int size) {
    file.write(static_cast<const char*>(static_cast<const void*>(data)),
               size * sizeof(T));
}
}  // namespace binary

#endif