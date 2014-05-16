include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(TxHPCG_TESTS
    test_CPCSR
    test_ComputeRestriction
    test_ComputeProlongation
    test_ComputeWAXPBY
   )
set(TxHPCG_PAR_TESTS
    test_ComputeSPMV
    test_ComputeSYMGS
    test_ComputeMG
    test_ComputeDotProduct
   )
foreach(t ${TxHPCG_TESTS} ${TxHPCG_PAR_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t}
      testUtils
      TxHPCG
      ${Gelus_LIBRARIES}
      ${CUDA_cusparse_LIBRARY}
      ${CUDA_LIBRARIES}
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


