#ifndef BINARY_CUH
#define BINARY_CUH

#include <iostream>

namespace binary {
template <typename T>
void ReadBinary(std::istream& file, T& data) {
    file.read(static_cast<char*>(static_cast<void*>(&data)), sizeof(data));
}

template <typename T>
void WriteBinary(std::ostream& file, const T& data) {
    file.write(static_cast<const char*>(static_cast<const void*>(&data)),
               sizeof(data));
}

template <typename T>
void ReadBinaryArray(std::istream& file, T* data, int size) {
    file.read(static_cast<char*>(static_cast<void*>(data)), size * sizeof(T));
}

template <typename T>
void WriteBinaryArray(std::ostream& file, const T* data, int size) {
    file.write(static_cast<const char*>(static_cast<const void*>(data)),
               size * sizeof(T));
}
}  // namespace binary

#endif