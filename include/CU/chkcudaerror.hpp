#ifndef CHKCUDAERROR_HPP
#define CHKCUDAERROR_HPP

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <gelusBase.h>
#include <iostream>
#include <stdexcept>

#define CHKCUDAERR(err) __chkCudaErr(err, __FILE__, __LINE__)
inline static cudaError_t __chkCudaErr(cudaError_t err, const char *file,
                                       const int line) {
  if (cudaSuccess != err) {
    std::cerr << file << "(" << line << "(): Cuda runtime API error (" << err
              << "):\n" << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("Fatal cuda error.");
  }
  return err;
}

#define CHKCUSPARSEERR(err) __chkCusparseErr(err, __FILE__, __LINE__)
inline static cusparseStatus_t __chkCusparseErr(cusparseStatus_t err,
                                                const char *file,
                                                const int line) {
  if (cudaSuccess != (cudaError_t)err) {
    std::cerr << file << "(" << line << "): Cusparse runtime API error ("
              << err << ")" << std::endl;
    throw std::runtime_error("Fatal cusparse error.");
  }
  return err;
}

#define CHKGELUSERR(err) __chkGelusErr(err, __FILE__, __LINE__)
inline static gelusStatus_t __chkGelusErr(gelusStatus_t err, const char *file,
                                          const int line) {
  if (GELUS_STATUS_SUCCESS != err) {
    std::cerr << file << "(" << line << "): Gelus runtime API error ("
              << err << ")" << std::endl;
    throw std::runtime_error("Fatal gelus error.");
  }
  return err;
}

#endif

