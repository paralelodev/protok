#include <protok.h>

using namespace protok;

int main() {
  // one-to-one addition of vectors

  int N = 1000;
  int A[1000];
  int B[1000];
  int C[1000];

  // data initialization goes here ...

  compute(
      {.BaseCU = ComputingUnity::CPU, .DistributionCU = ComputingUnity::THREAD},
      {.lowerbound = 0, .upperbound = N, .stride = 1, .type = RangeType::SPACE},
      [&A, &B, &C](int &i) { C[i] = A[i] + B[i]; });

  // one-to-one multiplication of vectors

  compute(Distributions::AccelOnTeams(), {0, N, 1},
          [&A, &B, &C](int &i) { A[i] = B[i] * C[i]; });

  // one-to-one multiplication of matrices

  int M = 100;
  int D[100][100];
  int E[100][100];
  int F[100][100];

  // data initialization goes here ...

  Range X = {0, M, 1, RangeType::SPACE};
  Range Y = {0, M, 1, RangeType::SPACE};
  compute(Distributions::CpuOnThreads(), X, Y,
          [&D, &E, &F](int &i, int &j) { F[i][j] = D[i][j] * E[i][j]; });

  // matrix multiplication

  int L = 100;
  int G[100][100];
  int H[100][100];
  int I[100][100];

  // data initialization goes here ...

  Range S = {0, L, 1, RangeType::SPACE};
  Range T = {0, L, 1, RangeType::SPACE};
  Range U = {0, L, 1, RangeType::DIMENSION};

  compute(
      Distributions::CpuOnThreads(), S, T, U,
      [&G, &H, &I](int &i, int &j, int &k) { I[i][j] += G[i][k] * H[k][j]; });
}
