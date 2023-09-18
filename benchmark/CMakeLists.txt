SET(OPENMP_FLAGS "-fopenmp")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OPENMP_FLAGS}")

add_executable(parallelfor parallelfor.cpp)

target_link_libraries(parallelfor
  PRIVATE
  protok
  )
