#!/bin/sh

PATH="/src_ivy/dmeiser/PTSOLVE/bin:/scr_ivy/dmeiser/PTSOLVE/openmpi/bin:$PATH" \
    /scr_ivy/dmeiser/PTSOLVE/cmake/bin/cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DENABLE_PARALLEL:BOOL=TRUE \
    -DSUPRA_SEARCH_PATH:PATH=/scr_ivy/dmeiser/PTSOLVE \
    -DCMAKE_C_COMPILER:FILEPATH=/scr_ivy/dmeiser/PTSOLVE/openmpi/bin/mpicc \
    -DCMAKE_CXX_COMPILER:FILEPATH=/scr_ivy/dmeiser/PTSOLVE/openmpi/bin/mpicxx \
    -DCMAKE_Fortran_COMPILER:FILEPATH='/scr_ivy/dmeiser/PTSOLVE/openmpi/bin/mpif90' \
    -DHPCG_SOURCE_DIR:PATH=/scr_ivy/dmeiser/HPCG/src \
    ..

