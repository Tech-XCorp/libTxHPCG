# CMakeLists.txt for TxHPCG, the Tx implementation of the HPCG benchmark
#
# variables:
#
# HPCG_SOURCE_DIR: location of the hpcg source directory
#

project(TxHPCG)
set(PROJECT_NAME "TxHPCG")
set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

cmake_minimum_required(VERSION 2.8.3)
enable_testing()
set(NOFORTRAN TRUE)
include(${PROJECT_SOURCE_DIR}/scimake/SciInit.cmake)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

option(ENABLE_MEMCHECK 
    "Enables testing to use valgrind to track memory leaks in tests."
    OFF)

# Packages
find_package(CUDA)
find_cuda_helper_libs(cusparse)
find_package(SciCusp)
find_package(SciOpenCL)
find_package(Gelus)
find_package(SciBoost COMPONENTS unit_test_framework REQUIRED)
find_package(SciMpi REQUIRED)
set(HAVE_MPI TRUE)

# Configuration of cuda flags
if (WIN32)
    set(CMAKE_CXX_FLAGS "-nologo -openmp -EHsc ${CMAKE_CXX_FLAGS}")
    set(CUDA_NVCC_FLAGS "--generate-code arch=compute_20,code=sm_20
   --generate-code arch=compute_20,code=sm_21 --generate-code
   arch=compute_30,code=sm_30 --generate-code
   arch=compute_35,code=sm_35 -use_fast_math -Xcompiler -openmp
   -Xcompiler -EHsc ${CUDA_NVCC_FLAGS}")
else ()
    set(CMAKE_CXX_FLAGS "-Wall -fPIC -fopenmp")
    set(CUDA_NVCC_FLAGS --generate-code arch=compute_20,code=sm_20
   --generate-code arch=compute_20,code=sm_21 --generate-code
   arch=compute_30,code=sm_30 --generate-code
   arch=compute_35,code=sm_35 -O3 --use_fast_math
   --compiler-options=-fPIC,-O3,-fopenmp)
endif ()

# Permissions
set(SCI_GROUP_WRITE GROUP_WRITE)
set(SCI_WORLD_FILE_PERMS WORLD_READ)
set(SCI_WORLD_PROGRAM_PERMS WORLD_READ WORLD_EXECUTE)

# use valgrind for memchecks
if (ENABLE_MEMCHECK AND NOT WIN32)
  find_program(VALGRIND_EXE valgrind)
  if (VALGRIND_EXE)
    set(CTEST_MEMORYCHECK_COMMAND ${VALGRIND_EXE})
  endif ()
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HPCG_SOURCE_DIR}
    ${Gelus_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    )
include_directories(SYSTEM 
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${MPI_INCLUDE_DIRS}
    )

configure_file(${CMAKE_SOURCE_DIR}/config-cmake.h.in config.h)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tests)


# Finalize
set(COMMON_INSTALL FALSE)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Tech-X implementation of HPCG benchmark.")
set(CONFIG_FILES)  # Clear out

include(${CMAKE_SOURCE_DIR}/scimake/SciFinalize.cmake)

