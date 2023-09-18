#include <protok.h>

using namespace protok;

int main() {
  int N = 1000;
  int A[1000] = {1};
  int B[1000] = {2};
  int C[1000] = {0};

  compute(
      {.BaseCU = ComputingUnity::CPU, .DistributionCU = ComputingUnity::THREAD},
      {.lowerbound = 0, .upperbound = N, .stride = 1},
      [&A, &B, &C](int &i) { C[i] = A[i] + B[i]; });

  compute(
      {.BaseCU = ComputingUnity::ACCEL, .DistributionCU = ComputingUnity::THREAD},
      {.lowerbound = 0, .upperbound = N, .stride = 1},
      [&A, &B, &C](int &i) { A[i] = B[i] * C[i]; });
}
