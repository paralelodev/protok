#include <omp.h>
#include <protok.h>
#include <vector>

using namespace protok;
using Vector = std::vector<int>;
using Matrix = std::vector<std::vector<int>>;

static void benchmarkProtok(Matrix A, Matrix B, Matrix C, int N) {
  double itime, ftime, exec_time;
  itime = omp_get_wtime();

  Range X, Y = {0, N, 1, RangeType::SPACE};
  Range Z = {0, N, 1, RangeType::DIMENSION};

  compute(
      Distributions::CpuOnThreads(), X, Y, Z,
      [&A, &B, &C](int &i, int &j, int &k) { C[i][j] += A[i][k] * B[k][j]; });

  ftime = omp_get_wtime();
  exec_time = ftime - itime;
  printf("ProtoK: %f  C[0][0]=%d\n", exec_time, C[0][0]);
}

static void benchmarkOpenMP(Matrix A, Matrix B, Matrix C, int N) {
  double itime, ftime, exec_time;
  itime = omp_get_wtime();

#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  ftime = omp_get_wtime();
  exec_time = ftime - itime;
  printf("OpenMP: %f  C[0][0]=%d\n", exec_time, C[0][0]);
}

static void benchmarkSerial(Matrix A, Matrix B, Matrix C, int N) {
  double itime, ftime, exec_time;
  itime = omp_get_wtime();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  ftime = omp_get_wtime();
  exec_time = ftime - itime;
  printf("Serial: %f  C[0][0]=%d\n", exec_time, C[0][0]);
}

int main() {
  int N = 10000;
  Matrix A(N, Vector(N, 1));
  Matrix B(N, Vector(N, 2));
  Matrix C(N, Vector(N, 0));
  benchmarkProtok(A, B, C, N);
  benchmarkOpenMP(A, B, C, N);
  benchmarkSerial(A, B, C, N);
}
