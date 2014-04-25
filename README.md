# libTxHPCG

Library with Tech-X's GPU accelerated HPCG kernels.

For more information on the HPCG benchmark see 
[the sandia HPCG website](https://software.sandia.gov/hpcg/default.php)


## Prerequisites

The following software packages and libraries are needed in order to
build and use libTxHPCG:

- HPCG

- CUDA

- GELUS

- MPI

- cmake (version 2.8.10 or higher)


## Getting Started

### Quick start

### Obtaining prerequisites

libTxHPCG uses `cmake` with some enhanced modules in Tech-X's scimake
modules. The scimake library can be optained from the 
[scimake repository on sourceforge](http://sourceforge.net/projects/scimake).
It should be installed in a subdirectory of the libTxHPCG repository
called `scimake`. On linux you can use the
[external_repos.sh](./external_repos.sh) script:
```
bash external_repos.sh
```


### Configuration

Configuration is done by means of cmake.  You need to tell cmake where
the HPCG src directory is located using the `HPCG_SRC_DIR` variable.
For example:

```
cmake \
  -DHPCG_SOURCE_DIR=../hpcg-2.1/src/ \
  -DSUPRA_SEARCH_PATH=${HOME}/software \
  -DCMAKE_CXX_COMPILER=${HOME}/software/openmpi/bin/mpicxx \
  -DCMAKE_C_COMPILER=${HOME/software/openmpi/bin/mpicc \
  -DMPI_CXX_COMPILER=${HOME}/software/openmpi/bin/mpicxx
  -DMPI_C_COMPILER=${HOME}/software/openmpi/bin/mpicc \
  -DMPIEXEC=${HOME}/software/openmpi/bin/mpirun \
  -DCMAKE_INSTALL_PREFIX=${HOME}/software/TxHPCG \
  ../libTxHPCG/ \
```

This assumes that `openmpi` has been installed in `${HOME}/software` and
the `hpcg` source dir is in `../hpcg-2.1/src/`.


### Building, testing, and installing

After successfully configuring `libTxHPCG` is built, tested, and
installed by means of:

```
make -j8
make test
make install
```

