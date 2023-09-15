#include <iostream>

namespace protok {
struct Space {
  int lowerbound;
  int upperbound;
  int stride;
};

enum Level { VECTOR, THREAD, ACCEL, RANK };

template <typename Kernel>
int compute(Level level, Space space, Kernel kernel) {
  switch (level) {
  case THREAD:
#pragma omp parallel for
    for (int i = space.lowerbound; i < space.upperbound; i += space.stride) {
      kernel(i);
    }
    break;
  case VECTOR:
#pragma omp simd
    for (int i = space.lowerbound; i < space.upperbound; i += space.stride) {
      kernel(i);
    }
    break;
  case ACCEL:
#pragma omp target teams distribute
    for (int i = space.lowerbound; i < space.upperbound; i += space.stride) {
      kernel(i);
    }
    break;
  case RANK:
    std::cerr << "Rank level parallelism not supported yet\n";
    exit(0);
  }

  return 0;
}
} // namespace protok
