#include <protok.h>

using namespace protok;

// one-to-one addition of vectors
static void vector_add() {
  int N = 1000;
  int A[1000];
  int B[1000];
  int C[1000];

  // data initialization goes here ...

  compute([&A, &B, &C](int &i) { C[i] = A[i] + B[i]; },
          {.lowerbound = 0,
           .upperbound = N,
           .stride = 1,
           .type = RangeType::SPACE});
}

// one-to-one multiplication of vectors
static void vector_mul() {
  int N = 1000;
  int A[1000];
  int B[1000];
  int C[1000];

  // data initialization goes here ...

  compute([&A, &B, &C](int &i) { A[i] = B[i] * C[i]; },
          {0, N, 1, RangeType::SPACE}, Distributions::AccelOnTeams());
}

// one-to-one multiplication of matrices
static void matrix_mul_naive() {
  int N = 100;
  int A[100][100];
  int B[100][100];
  int C[100][100];

  // data initialization goes here ...

  Range X = {0, N, 1, RangeType::SPACE};
  Range Y = {0, N, 1, RangeType::SPACE};
  compute([&A, &B, &C](int &i, int &j) { C[i][j] = A[i][j] * B[i][j]; }, X, Y);
}

// matrix multiplication
static void matrix_mul() {
  int N = 100;
  int A[100][100];
  int B[100][100];
  int C[100][100];

  // data initialization goes here ...

  Range X = {0, N, 1, RangeType::SPACE};
  Range Y = {0, N, 1, RangeType::SPACE};
  Range Z = {0, N, 1, RangeType::DIMENSION};

  compute(
      [&A, &B, &C](int &i, int &j, int &k) { C[i][j] += A[i][k] * B[k][j]; }, X,
      Y, Z);
}

int main() {
  vector_add();
  vector_mul();
  matrix_mul_naive();
  matrix_mul();
}
