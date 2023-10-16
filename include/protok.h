#include "macros.h"
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
int compute(ComputingDistribution distribution, Range outerrange,
            Kernel kernel) {
  if (outerrange.type != RangeType::SPACE) {
    std::cerr << "The range must be of space type\n";
    exit(0);
  }

  switch (distribution.BaseCU) {
  case ComputingUnity::CPU:
    switch (distribution.DistributionCU) {
    case ComputingUnity::THREAD:
      PARALLELFOR_1D(outerrange)
      break;
    case ComputingUnity::VECTOR:
      SIMD_1D(outerrange)
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
      TARGET_1D(outerrange)
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
            Range middlerange, Kernel kernel) {
  if (outerrange.type != RangeType::SPACE) {
    std::cerr << "The outer range must be of space type\n";
    exit(0);
  }

  // Set the collapse level
  int collapseLevel = middlerange.type == RangeType::DIMENSION ? 1 : 2;

  switch (distribution.BaseCU) {
  case ComputingUnity::CPU:
    switch (distribution.DistributionCU) {
    case ComputingUnity::THREAD:
      switch (collapseLevel) {
      case 2:
        PARALLELFOR_2D_C2(outerrange, middlerange)
        break;
      default:
        PARALLELFOR_2D(outerrange, middlerange)
        break;
      }
      break;
    case ComputingUnity::VECTOR:
      switch (collapseLevel) {
      case 2:
        SIMD_2D_C2(outerrange, middlerange)
        break;
      default:
        SIMD_2D(outerrange, middlerange)
        break;
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
      switch (collapseLevel) {
      case 2:
        TARGET_2D_C2(outerrange, middlerange)
        break;
      default:
        TARGET_2D(outerrange, middlerange)
        break;
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
  if (outerrange.type != RangeType::SPACE) {
    std::cerr << "The outer range must be of space type\n";
    exit(0);
  }

  // Set the collapse level
  int collapseLevel = 1;
  if (middlerange.type == RangeType::SPACE) {
    collapseLevel++;
  }
  if (innerrange.type == RangeType::SPACE) {
    if (middlerange.type == RangeType::DIMENSION) {
      std::cerr << "The inner range must be of dimension type too\n";
      exit(0);
    }
    collapseLevel++;
  }

  switch (distribution.BaseCU) {
  case ComputingUnity::CPU:
    switch (distribution.DistributionCU) {
    case ComputingUnity::THREAD:
      switch (collapseLevel) {
      case 2:
        PARALLELFOR_3D_C2(outerrange, middlerange, innerrange)
        break;
      case 3:
        PARALLELFOR_3D_C3(outerrange, middlerange, innerrange)
        break;
      default:
        PARALLELFOR_3D(outerrange, middlerange, innerrange)
        break;
      }
      break;
    case ComputingUnity::VECTOR:
      switch (collapseLevel) {
      case 2:
        SIMD_3D_C2(outerrange, middlerange, innerrange)
        break;
      case 3:
        SIMD_3D_C3(outerrange, middlerange, innerrange)
        break;
      default:
        SIMD_3D(outerrange, middlerange, innerrange)
        break;
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
      switch (collapseLevel) {
      case 2:
        TARGET_3D_C2(outerrange, middlerange, innerrange)
        break;
      case 3:
        TARGET_3D_C3(outerrange, middlerange, innerrange)
        break;
      default:
        TARGET_3D(outerrange, middlerange, innerrange)
        break;
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
