cmake_minimum_required(VERSION 3.0)
# Time-dependent Lanczos solver for real and imaginary time propagation
project(E4X4 C CXX Fortran)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Enforce to use C++11
include(CXXStandard)
use_cxx11()

# ALPSCore library
find_package(ALPSCore COMPONENTS params hdf5 gf REQUIRED)
# implicitly restarted ARnoldi PACKage
find_package(ARPACK REQUIRED)
# BLAS library
find_package(BLAS REQUIRED)
# LAPACK library
find_package(LAPACK REQUIRED)
# Eigen3 support
find_package(Eigen3 REQUIRED)
# Exact Diagonalization library
find_package(EDLib REQUIRED)
# Add library includes
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${ALPSCore_INCLUDES})
include_directories(${EDLIB_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EDLib_INCLUDE_DIR})

# keep debug symbols always
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# add specific release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_DISABLE_ASSERTS -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m64")

# external libraries
set(extlibs
    ${Boost_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${ALPSCore_LIBRARIES}
    ${PARPACK_LIB}
    ${ARPACK_LIB}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    )

# if we want to have mpi we need to check whether ALPSCore was compiled with MPI support
# and check that we do have MPI libraries
if(USE_MPI)
    if(!ALPS_HAVE_MPI)
        MESSAGE(FATAL_ERROR "Enable MPI support in ALPSCore.")
    endif(!ALPS_HAVE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        # Propagate USE_MPI flag into prepocessor
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")
    endif(MPI_FOUND)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${MPI_Fortran_COMPILE_FLAG}")
    set(extlibs ${extlibs}
                ${MPI_CXX_LIBRARIES}
                ${MPI_Fortran_LIBRARIES})
endif(USE_MPI)

# add includes
include_directories(${E4X4_SOURCE_DIR}/include)
# add includes as sub-directory so IDE can index it
add_subdirectory(include/edlib-4x4)

# Unit and acceptence tests
if(Testing)
    include(EnableGtests)
    include_directories(${GTEST_INCLUDE_DIR})
    enable_testing()
    add_subdirectory(test)
    include_directories(${E4X4_SOURCE_DIR}/test)
endif(Testing)

# main Time-Dependent Lanczos code
set(SOURCE_FILES main.cpp)
add_executable(Hubbard ${SOURCE_FILES})
# add link libraries
target_link_libraries(Hubbard ${extlibs})
