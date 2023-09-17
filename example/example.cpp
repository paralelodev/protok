#include <protok.h>

int main() {
  int N = 1000;
  int A[1000] = {1};
  int B[1000] = {2};
  int C[1000] = {0};

  protok::compute({.BaseCU = protok::ComputingUnity::CPU,
                   .DistributionCU = protok::ComputingUnity::THREAD},
                  {.lowerbound = 0, .upperbound = N, .stride = 1},
                  [&A, &B, &C](int &i) { C[i] = A[i] + B[i]; });
}
