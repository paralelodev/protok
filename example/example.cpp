#include <protok.h>

using namespace protok;

int main() {
  int N = 1000;
  int A[1000];
  int B[1000];
  int C[1000];

  Space I = {.lowerbound = 0, .upperbound = N, .stride = 1};

  compute(Distributions::CpuOnThreads(), I,
          [&A, &B, &C](int &i) { C[i] = A[i] + B[i]; });

  compute(Distributions::AccelOnTeams(),
          {.lowerbound = 0, .upperbound = N, .stride = 1},
          [&A, &B, &C](int &i) { A[i] = B[i] * C[i]; });

  int M = 100;
  int D[100][100];
  int E[100][100];
  int F[100][100];

  Space X, Y = {.lowerbound = 0, .upperbound = M, .stride = 1};

  compute(Distributions::CpuOnThreads(), X, Y,
          [&D, &E, &F](int &i, int &j) { F[i][j] = D[i][j] * E[i][j]; });
}
