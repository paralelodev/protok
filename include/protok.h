namespace protok {
struct Space {
  int lowerbound;
  int upperbound;
  int stride;
};

template <typename Kernel> int compute(Space space, Kernel kernel) {
#pragma omp parallel for
  for (int i = space.lowerbound; i < space.upperbound; i += space.stride) {
    kernel(i);
  }

  return 0;
}
} // namespace protok
