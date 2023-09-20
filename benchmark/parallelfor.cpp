#include <omp.h>
#include <protok.h>
#include <vector>

using namespace protok;

static void benchmarkProtok(std::vector<int> A, std::vector<int> B,
                            std::vector<int> C, int N) {
  double itime, ftime, exec_time;
  itime = omp_get_wtime();

  compute(Distributions::CpuOnThreads(), {0, N, 1},
          [&A, &B, &C](int &i) { C[i] = A[i] + B[i]; });

  ftime = omp_get_wtime();
  exec_time = ftime - itime;
  printf("ProtoK: %f  C[0]=%d\n", exec_time, C[0]);
}

static void benchmarkOpenMP(std::vector<int> A, std::vector<int> B,
                            std::vector<int> C, int N) {
  double itime, ftime, exec_time;
  itime = omp_get_wtime();

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }

  ftime = omp_get_wtime();
  exec_time = ftime - itime;
  printf("OpenMP: %f  C[0]=%d\n", exec_time, C[0]);
}

static void benchmarkSerial(std::vector<int> A, std::vector<int> B,
                            std::vector<int> C, int N) {
  double itime, ftime, exec_time;
  itime = omp_get_wtime();

  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }

  ftime = omp_get_wtime();
  exec_time = ftime - itime;
  printf("Serial: %f  C[0]=%d\n", exec_time, C[0]);
}

int main() {
  int N = 1000000000;
  std::vector<int> A(N, 1);
  std::vector<int> B(N, 2);
  std::vector<int> C(N, 0);
  benchmarkProtok(A, B, C, N);
  benchmarkOpenMP(A, B, C, N);
  benchmarkSerial(A, B, C, N);
}
