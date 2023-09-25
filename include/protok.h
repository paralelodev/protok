#include <iostream>

namespace protok {

enum class RangeType { SPACE, DIMENSION };

struct Range {
  int lowerbound = 0;
  int upperbound = 0;
  int stride = 1;
  RangeType type = RangeType::SPACE;
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
int compute(ComputingDistribution distribution, Range range, Kernel kernel) {
  switch (distribution.BaseCU) {
  case ComputingUnity::CPU:
    switch (distribution.DistributionCU) {
    case ComputingUnity::THREAD:
#pragma omp parallel for
      for (int i = range.lowerbound; i < range.upperbound; i += range.stride) {
        kernel(i);
      }
      break;
    case ComputingUnity::VECTOR:
#pragma omp simd
      for (int i = range.lowerbound; i < range.upperbound; i += range.stride) {
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
      for (int i = range.lowerbound; i < range.upperbound; i += range.stride) {
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
int compute(ComputingDistribution distribution, Range outerrange,
            Range innerrange, Kernel kernel) {
  // TODO: Set the collapse level
  int collapseLevel = 2;
  if (innerrange.type == RangeType::DIMENSION) {
    collapseLevel--;
  }

  switch (distribution.BaseCU) {
  case ComputingUnity::CPU:
    switch (distribution.DistributionCU) {
    case ComputingUnity::THREAD:
#pragma omp parallel for
      for (int i = outerrange.lowerbound; i < outerrange.upperbound;
           i += outerrange.stride) {
        for (int j = innerrange.lowerbound; j < innerrange.upperbound;
             j += innerrange.stride) {
          kernel(i, j);
        }
      }
      break;
    case ComputingUnity::VECTOR:
#pragma omp simd 
      for (int i = outerrange.lowerbound; i < outerrange.upperbound;
           i += outerrange.stride) {
        for (int j = innerrange.lowerbound; j < innerrange.upperbound;
             j += innerrange.stride) {
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
#pragma omp target teams distribute 
      for (int i = outerrange.lowerbound; i < outerrange.upperbound;
           i += outerrange.stride) {
        for (int j = innerrange.lowerbound; j < innerrange.upperbound;
             j += innerrange.stride) {
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

template <typename Kernel>
int compute(ComputingDistribution distribution, Range outerrange,
            Range middlerange, Range innerrange, Kernel kernel) {
  // Set the collapse level
  int collapseLevel = 3;
  if (middlerange.type == RangeType::DIMENSION) {
    if (innerrange.type != RangeType::DIMENSION) {
      std::cerr << "The inner range must be of dimension type too\n";
      exit(0);
    } else {
      collapseLevel--;
    }
    collapseLevel--;
  } else if (innerrange.type == RangeType::DIMENSION) {
    collapseLevel--;
  }

  switch (distribution.BaseCU) {
  case ComputingUnity::CPU:
    switch (distribution.DistributionCU) {
    case ComputingUnity::THREAD:
#pragma omp parallel for 
      for (int i = outerrange.lowerbound; i < outerrange.upperbound;
           i += outerrange.stride) {
        for (int j = middlerange.lowerbound; j < middlerange.upperbound;
             j += middlerange.stride) {
          for (int k = innerrange.lowerbound; k < innerrange.upperbound;
               k += innerrange.stride) {
            kernel(i, j, k);
          }
        }
      }
      break;
    case ComputingUnity::VECTOR:
#pragma omp simd 
      for (int i = outerrange.lowerbound; i < outerrange.upperbound;
           i += outerrange.stride) {
        for (int j = middlerange.lowerbound; j < middlerange.upperbound;
             j += middlerange.stride) {
          for (int k = innerrange.lowerbound; k < innerrange.upperbound;
               k += innerrange.stride) {
            kernel(i, j, k);
          }
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
#pragma omp target teams distribute 
      for (int i = outerrange.lowerbound; i < outerrange.upperbound;
           i += outerrange.stride) {
        for (int j = middlerange.lowerbound; j < middlerange.upperbound;
             j += middlerange.stride) {
          for (int k = innerrange.lowerbound; k < innerrange.upperbound;
               k += innerrange.stride) {
            kernel(i, j, k);
          }
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
