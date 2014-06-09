include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(testUtils 
    testUtils.cpp 
    MPIFixture.cpp 
    HPCGFixture.cpp
    ${HPCG_SOURCE_DIR}/SetupHalo.cpp
    ${HPCG_SOURCE_DIR}/ExchangeHalo.cpp
    ${HPCG_SOURCE_DIR}/init.cpp
    ${HPCG_SOURCE_DIR}/ReadHpcgDat.cpp
    ${HPCG_SOURCE_DIR}/ComputeWAXPBY_ref.cpp
    ${HPCG_SOURCE_DIR}/ComputeDotProduct_ref.cpp
    ${HPCG_SOURCE_DIR}/mytimer.cpp
    )
target_link_libraries(
    testUtils 
    TxHPCG
    )
set(TxHPCG_TESTS
    test_TestUtils
    test_TxMatrixOptimizationDataBase
   )
set(TxHPCG_PAR_TESTS
    test_HPCGFixture
    test_MPI
    test_MPIFixture
   )
foreach(t ${TxHPCG_TESTS} ${TxHPCG_PAR_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t}
      testUtils
      TxHPCG
      ${Boost_LIBRARIES}
      )
endforeach()
foreach(t ${TxHPCG_TESTS})
  add_test(${t} ${t})
endforeach()
foreach(t ${TxHPCG_PAR_TESTS})
  add_test(NAME "${t}_1_proc" COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${t})
  add_test(NAME "${t}_2_procs" COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${t})
endforeach()
if (ENABLE_MEMCHECK)
  add_test(NAME "memtest_HPCGFixture_1_proc" COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${VALGRIND_EXE} ./test_HPCGFixture)
  add_test(NAME "memtest_HPCGFixture_2_proc" COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${VALGRIND_EXE} ./test_HPCGFixture)
  add_test(NAME "memtest_MPIFixture_1_proc" COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${VALGRIND_EXE} ./test_MPIFixture)
  add_test(NAME "memtest_MPIFixture_2_proc" COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${VALGRIND_EXE} ./test_MPIFixture)
endif ()

if(BUILD_CU_TXHPCG)
  add_subdirectory(CU)
endif()
if(BUILD_OCL_TXHPCG)
  add_subdirectory(OCL)
endif()

