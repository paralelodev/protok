#include <protok.h>

using namespace protok;

int main() {
  int N = 1000;
  int A[1000];
  int B[1000];
  int C[1000];

  // data initialization goes here ...

  compute(
      {.BaseCU = ComputingUnity::CPU, .DistributionCU = ComputingUnity::THREAD},
      {.lowerbound = 0, .upperbound = N, .stride = 1},
      [&A, &B, &C](int &i) { C[i] = A[i] + B[i]; });

  compute(Distributions::AccelOnTeams(), {0, N, 1},
          [&A, &B, &C](int &i) { A[i] = B[i] * C[i]; });

  int M = 100;
  int D[100][100];
  int E[100][100];
  int F[100][100];

  // data initialization goes here ...

  Space X, Y = {0, M, 1};
  compute(Distributions::CpuOnThreads(), X, Y,
          [&D, &E, &F](int &i, int &j) { F[i][j] = D[i][j] * E[i][j]; });

  // matrix multiplication

  int L = 100;
  int G[100][100];
  int H[100][100];
  int I[100][100];

  // data initialization goes here ...

  Space S, T, U = {0, L, 1};
  compute(
      Distributions::CpuOnThreads(), S, T, U,
      [&G, &H, &I](int &i, int &j, int &k) { I[i][j] += G[i][k] * H[k][j]; });
}
