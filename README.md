# ProtoK

ProtoK is a library for quickly prototyping parallel kernels. 
It uses OpenMP in the background to execute the code in parallel.

The idea is simple: do not write loops, just configure the parallel environment and provide the code instructions that must be executed in parallel.
E.g.:

```c++
#include <protok.h>

int main() {
  int N = 1000;
  int A[1000] = {1};
  int B[1000] = {2};
  int C[1000] = {0};

  protok::compute(protok::VECTOR,
                  {.lowerbound = 0, .upperbound = N, .stride = 1},
                  [&A, &B, &C](int &i) { C[i] = A[i] + B[i]; });
}
```
