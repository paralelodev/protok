# ProtoK

ProtoK is a C++ library for quickly prototyping parallel kernels. 
It uses OpenMP in the background to execute the code in parallel.

The idea is simple: do not write loops, just configure the parallel environment and provide the code instructions that must be executed in parallel.
E.g.:

```c++
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
      {.lowerbound = 0, .upperbound = N, .stride = 1, .type = RangeType::SPACE},
      [&A, &B, &C](int &i) { C[i] = A[i] + B[i]; });
}
```

The development of this library serves as an experimentation ground for the design of the **Ptk** language. 
See the [example/example.ptk](https://github.com/paralelodev/protok/blob/main/example/example.ptk) file for a preview of the language.

Feel free to contact the autor if you wish to know more and maybe contribute: [paralelodevinfo@gmail.com](paralelodevinfo@gmail.com)
