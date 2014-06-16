######################################################################
# FindGelus: Module to find include directories and libraries
#   for Gelus.
#
# This module will define the following variables:
#  HAVE_GELUS         = Whether have the Gelus library
#  Gelus_INCLUDE_DIRS = Location of Gelus includes
#  Gelus_LIBRARY_DIRS = Location of Gelus libraries
#  Gelus_LIBRARIES    = Required libraries
######################################################################

SciFindPackage(PACKAGE "Gelus"
              INSTALL_DIR "gelus"
              HEADERS "cuGelus.h"
              LIBRARIES "cugelus;gelusbase"
              )
message(STATUS "Gelus_LIBRARIES==${Gelus_LIBRARIES}")

if (GELUS_FOUND)
  message(STATUS "Found Gelus")
  set(HAVE_GELUS 1 CACHE BOOL "Whether have the GELUS library")
else ()
  message(STATUS "Did not find GELUS. Use -DGELUS_DIR to specify the installation directory.")
  if (TxGelus_FIND_REQUIRED)
    message(FATAL_ERROR "Failed.")
  endif ()
endif ()

