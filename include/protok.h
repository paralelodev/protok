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

struct Distributions {
  static ComputingDistribution CpuOnThreads() {
    return {.BaseCU = ComputingUnity::CPU,
            .DistributionCU = ComputingUnity::THREAD};
  }

  static ComputingDistribution AccelOnTeams() {
    return {.BaseCU = ComputingUnity::ACCEL,
            .DistributionCU = ComputingUnity::TEAM};
  }
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

template <typename Kernel>
int compute(ComputingDistribution distribution, Space outerspace,
            Space innerspace, Kernel kernel) {
  switch (distribution.BaseCU) {
  case ComputingUnity::CPU:
    switch (distribution.DistributionCU) {
    case ComputingUnity::THREAD:
#pragma omp parallel for collapse(2)
      for (int i = outerspace.lowerbound; i < outerspace.upperbound;
           i += outerspace.stride) {
        for (int j = innerspace.lowerbound; j < innerspace.upperbound;
             j += innerspace.stride) {
          kernel(i, j);
        }
      }
      break;
    case ComputingUnity::VECTOR:
#pragma omp simd collapse(2)
      for (int i = outerspace.lowerbound; i < outerspace.upperbound;
           i += outerspace.stride) {
        for (int j = innerspace.lowerbound; j < innerspace.upperbound;
             j += innerspace.stride) {
          kernel(i, j);
        }
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
#pragma omp target teams distribute collapse(2)
      for (int i = outerspace.lowerbound; i < outerspace.upperbound;
           i += outerspace.stride) {
        for (int j = innerspace.lowerbound; j < innerspace.upperbound;
             j += innerspace.stride) {
          kernel(i, j);
        }
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
