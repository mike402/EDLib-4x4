include_directories(${E4X4_SOURCE_DIR}/include)
include_directories(${E4X4_SOURCE_DIR}/lib)

add_executable(DensityMatrixTest DensityMatrix_Test.cpp)

target_link_libraries(DensityMatrixTest ${extlibs} ${parlibs} ${GTEST_LIBRARY})

file(COPY input DESTINATION ${CMAKE_BINARY_DIR}/test)
