#include <iostream>

namespace protok {
struct Space {
  int lowerbound;
  int upperbound;
  int stride;
};

enum class ComputingUnity { NODE, CPU, CORE, ACCEL, TEAM, THREAD, VECTOR };

struct ComputingDistribution {
  ComputingUnity BaseCU;
  ComputingUnity DistributionCU;
};

template <typename Kernel>
int compute(ComputingDistribution distribution, Space space, Kernel kernel) {
  switch (distribution.BaseCU) {
  case ComputingUnity::CPU:
    switch (distribution.DistributionCU) {
    case ComputingUnity::THREAD:
#pragma omp parallel for
      for (int i = space.lowerbound; i < space.upperbound; i += space.stride) {
        kernel(i);
      }
      break;
    case ComputingUnity::VECTOR:
#pragma omp simd
      for (int i = space.lowerbound; i < space.upperbound; i += space.stride) {
        kernel(i);
      }
      break;
    default:
      std::cerr << "The provided level of parallelism inside a CPU is not "
                   "supported yet\n";
      exit(0);
    }
    break;

  case ComputingUnity::ACCEL:
    switch (distribution.DistributionCU) {
    case ComputingUnity::TEAM:
#pragma omp target teams distribute
      for (int i = space.lowerbound; i < space.upperbound; i += space.stride) {
        kernel(i);
      }
      break;
    default:
      std::cerr
          << "The provided level of parallelism inside an accelerator is not "
             "supported yet\n";
      exit(0);
    }
    break;

  default:
    std::cerr << "The provided base computing unity is not supported yet\n";
    exit(0);
  }

  return 0;
}
} // namespace protok
