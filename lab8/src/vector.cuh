#ifndef VECTOR_CUH
#define VECTOR_CUH

namespace vector {
template <typename T>
struct Vector3 {
    T x, y, z;
};

using Vector3i = Vector3<int>;
}  // namespace vector

#endif