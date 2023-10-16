#define PARALLELFOR _Pragma("omp parallel for firstprivate(outerrange)")
#define SIMD _Pragma("omp simd")
#define TARGET _Pragma("omp target teams distribute firstprivate(outerrange)")

#define PARALLELFOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) firstprivate(outerrange, middlerange)")
#define SIMD_COLLAPSE2 _Pragma("omp simd collapse(2)")
#define TARGET_COLLAPSE2 _Pragma("omp target teams distribute collapse(2) firstprivate(outerrange, middlerange)")

#define PARALLELFOR_COLLAPSE3 _Pragma("omp parallel for collapse(3) firstprivate(outerrange, middlerange, innerrange)")
#define SIMD_COLLAPSE3 _Pragma("omp simd collapse(3)")
#define TARGET_COLLAPSE3 _Pragma("omp target teams distribute collapse(3) firstprivate(outerrange, middlerange, innerrange)")

#define KERNEL_1D(i)                                                           \
  { kernel(i); }

#define KERNEL_2D(i, j)                                                        \
  { kernel(i, j); }

#define KERNEL_3D(i, j, k)                                                     \
  { kernel(i, j, k); }

#define OUTER_START(outerrange)                                                \
  for (int i = outerrange.lowerbound; i < outerrange.upperbound;               \
       i += outerrange.stride)
#define MIDDLE_START(middlerange)                                              \
  for (int j = middlerange.lowerbound; j < middlerange.upperbound;             \
       j += middlerange.stride)
#define INNER_START(innerrange)                                                \
  for (int k = innerrange.lowerbound; k < innerrange.upperbound;               \
       k += innerrange.stride)

#define PARALLELFOR_1D(outerrange)                                             \
  PARALLELFOR                                                                  \
  OUTER_START(outerrange)                                                      \
  KERNEL_1D(i)
#define SIMD_1D(outerrange) SIMD OUTER_START(outerrange) KERNEL_1D(i)
#define TARGET_1D(outerrange)                                                  \
  TARGET                                                                       \
  OUTER_START(outerrange)                                                      \
  KERNEL_1D(i)

#define PARALLELFOR_2D(outerrange, middlerange)                                \
  PARALLELFOR                                                                  \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  KERNEL_2D(i, j)
#define SIMD_2D(outerrange, middlerange)                                       \
  SIMD OUTER_START(outerrange) MIDDLE_START(middlerange) KERNEL_2D(i, j)
#define TARGET_2D(outerrange, middlerange)                                     \
  TARGET                                                                       \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  KERNEL_2D(i, j)

#define PARALLELFOR_2D_C2(outerrange, middlerange)                             \
  PARALLELFOR_COLLAPSE2                                                        \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  KERNEL_2D(i, j)
#define SIMD_2D_C2(outerrange, middlerange)                                    \
  SIMD_COLLAPSE2                                                               \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  KERNEL_2D(i, j)
#define TARGET_2D_C2(outerrange, middlerange)                                  \
  TARGET_COLLAPSE2                                                             \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  KERNEL_2D(i, j)

#define PARALLELFOR_3D(outerrange, middlerange, innerrange)                    \
  PARALLELFOR                                                                  \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  INNER_START(innerrange)                                                      \
  KERNEL_3D(i, j, k)
#define SIMD_3D(outerrange, middlerange, innerrange)                           \
  SIMD OUTER_START(outerrange) MIDDLE_START(middlerange)                       \
      INNER_START(innerrange) KERNEL_3D(i, j, k)
#define TARGET_3D(outerrange, middlerange, innerrange)                         \
  TARGET                                                                       \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  INNER_START(innerrange)                                                      \
  KERNEL_3D(i, j, k)

#define PARALLELFOR_3D_C2(outerrange, middlerange, innerrange)                 \
  PARALLELFOR_COLLAPSE2                                                        \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  INNER_START(innerrange)                                                      \
  KERNEL_3D(i, j, k)
#define SIMD_3D_C2(outerrange, middlerange, innerrange)                        \
  SIMD_COLLAPSE2                                                               \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  INNER_START(innerrange)                                                      \
  KERNEL_3D(i, j, k)
#define TARGET_3D_C2(outerrange, middlerange, innerrange)                      \
  TARGET_COLLAPSE2                                                             \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  INNER_START(innerrange)                                                      \
  KERNEL_3D(i, j, k)

#define PARALLELFOR_3D_C3(outerrange, middlerange, innerrange)                 \
  PARALLELFOR_COLLAPSE3                                                        \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  INNER_START(innerrange)                                                      \
  KERNEL_3D(i, j, k)
#define SIMD_3D_C3(outerrange, middlerange, innerrange)                        \
  SIMD_COLLAPSE3                                                               \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  INNER_START(innerrange)                                                      \
  KERNEL_3D(i, j, k)
#define TARGET_3D_C3(outerrange, middlerange, innerrange)                      \
  TARGET_COLLAPSE3                                                             \
  OUTER_START(outerrange)                                                      \
  MIDDLE_START(middlerange)                                                    \
  INNER_START(innerrange)                                                      \
  KERNEL_3D(i, j, k)
