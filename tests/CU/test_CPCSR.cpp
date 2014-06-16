#include "CPCSR.hpp"

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <CU/chkcudaerror.hpp>

int constructorTest();
int findDupTest();
int mergeTest();
int addEntryTest();
int getCPCSRArraysTest();
int spmvTest();

static const double EPS_D = 1.0e-16;

bool operator==(const ColValPair& a, const ColValPair& b) {
  return (a.col == b.col && fabs(a.val - b.val) < EPS_D);
}
void mkUnitVector(std::vector<double>* v, local_int_t n) {
  std::fill(v->begin(), v->end(), 0.0);
  v->at(n) = 1.0;
}
double computeMatrixElement(CPCSR& matrix, local_int_t m, local_int_t n) {
  std::vector<double> x(matrix.getNumCols());
  std::vector<double> y(matrix.getNumRows(), 0);
  double* x_d;
  CHKCUDAERR(cudaMalloc((void**)&x_d, x.size() * sizeof(x[0])));
  double* y_d;
  CHKCUDAERR(cudaMalloc((void**)&y_d, y.size() * sizeof(y[0])));

  mkUnitVector(&x, n);
  CHKCUDAERR(
      cudaMemcpy(x_d, &x[0], x.size() * sizeof(x[0]), cudaMemcpyHostToDevice));
  CHKCUDAERR(
      cudaMemcpy(y_d, &y[0], y.size() * sizeof(y[0]), cudaMemcpyHostToDevice));

  matrix.spmv(x_d, y_d);

  CHKCUDAERR(
      cudaMemcpy(&x[0], x_d, x.size() * sizeof(x[0]), cudaMemcpyDeviceToHost));
  CHKCUDAERR(
      cudaMemcpy(&y[0], y_d, y.size() * sizeof(y[0]), cudaMemcpyDeviceToHost));

  CHKCUDAERR(cudaFree(x_d));
  CHKCUDAERR(cudaFree(y_d));

  return y[m];
}

int main(int argn, char** argv) {
  int errs = 0;
  int newErrs = 0;
  newErrs = constructorTest();
  if (newErrs != 0) {
    std::cout << "constructorTest failed. " << newErrs << " errors occured.\n";
    errs += newErrs;
  }
  newErrs = findDupTest();
  if (newErrs != 0) {
    std::cout << "findDupTest failed. " << newErrs << " errors occured.\n";
    errs += newErrs;
  }
  newErrs = mergeTest();
  if (newErrs != 0) {
    std::cout << "mergeTest failed. " << newErrs << " errors occured.\n";
    errs += newErrs;
  }
  newErrs = addEntryTest();
  if (newErrs != 0) {
    std::cout << "addEntryTest failed. " << newErrs << " errors occured.\n";
    errs += newErrs;
  }
  newErrs = getCPCSRArraysTest();
  if (newErrs != 0) {
    std::cout << "getCPCSRArraysTest failed. " << newErrs << " errors occured.\n";
    errs += newErrs;
  }
  newErrs = spmvTest();
  if (newErrs != 0) {
    std::cout << "spmvTest failed. " << newErrs << " errors occured.\n";
    errs += newErrs;
  }
  return errs;
}
int constructorTest() {
  CPCSR a(10, 11);
  CPCSR b(a);
  CPCSR c = b;
  c = a;
  return 0;
}
int findDupTest() {
  int errs = 0;
  ColValPair ja(0, 0);
  std::vector<ColValPair> v;
  std::vector<ColValPair>::iterator dup = findDup(ja, v);
  if (dup != v.end()) ++errs;

  v.push_back(ColValPair(1, 0));
  dup = findDup(ja, v);
  if (dup != v.end()) ++errs;

  v.push_back(ColValPair(3, 0));
  dup = findDup(ja, v);
  if (dup != v.end()) ++errs;

  dup = findDup(ColValPair(1, 0), v);
  if (dup == v.end()) ++errs;
  if (std::distance(v.begin(), dup) != 0) errs += 1;

  return errs;
}
int mergeTest() {
  int errs = 0;

  // Merge an empty container
  std::vector<ColValPair> v;
  std::vector<ColValPair> w;
  merge(&v, w);
  if (v.empty() == false) ++errs;

  // Merge into an empty container
  w.push_back(ColValPair(0, 0));
  merge(&v, w);
  std::vector<ColValPair> expectedResult(1, ColValPair(0, 0));
  if (v.size() != 1) ++errs;
  if (!std::equal(v.begin(), v.end(), expectedResult.begin())) ++errs;

  // Merge same entry again
  merge(&v, w);
  if (v.size() != 1) ++errs;
  if (!std::equal(v.begin(), v.end(), expectedResult.begin())) ++errs;

  // Merge a new entry
  w.assign(1, ColValPair(1, 0));
  merge(&v, w);
  if (v.size() != 2) ++errs;
  expectedResult.push_back(ColValPair(1, 0));
  if (!std::equal(v.begin(), v.end(), expectedResult.begin())) ++errs;

  // Merge an old entry 
  w.assign(1, ColValPair(0, 2));
  merge(&v, w);
  if (v.size() != 2) ++errs;
  expectedResult[0] = ColValPair(0, 2);
  if (!std::equal(v.begin(), v.end(), expectedResult.begin())) ++errs;

  // Merge several entries
  w.assign(1, ColValPair(0, 2));
  w.push_back(ColValPair(5, 2.0));
  w.push_back(ColValPair(1, 3.0));
  merge(&v, w);
  if (v.size() != 3) ++errs;
  expectedResult.resize(3);
  expectedResult[0] = ColValPair(0, 4);
  expectedResult[1] = ColValPair(1, 3);
  expectedResult[2] = ColValPair(5, 2);
  if (!std::equal(v.begin(), v.end(), expectedResult.begin())) ++errs;

  return errs;
}
int addEntryTest() {
  int errs = 0;
  CPCSR a(7, 4);
  a.addEntry(0, 0, 1);
  if (a.getNNZ() != 1) ++errs;
  if (a.getNumNonZeroRows() != 1) ++errs;

  a.addEntry(0, 0, 2);
  if (a.getNumNonZeroRows() != 1) ++errs;

  a.addEntry(0, 2, 2);
  if (a.getNNZ() != 2) ++errs;
  if (a.getNumNonZeroRows() != 1) ++errs;

  a.addEntry(2, 0, 2);
  if (a.getNNZ() != 3) ++errs;
  if (a.getNumNonZeroRows() != 2) ++errs;

  return errs;
}
int getCPCSRArraysTest() {
  int errs = 0;
  CPCSR matrix(5, 14);
  std::vector<local_int_t> r;
  std::vector<local_int_t> o;
  std::vector<local_int_t> c;
  std::vector<double> a;

  matrix.getCPCSRArrays(&r, &o, &c, &a);
  if (!r.empty()) ++errs;
  if (o.size() != 1) ++errs;
  if (!c.empty()) ++errs;
  if (!r.empty()) ++errs;
  if (!a.empty()) ++errs;

  matrix.addEntry(3, 0, 4);
  matrix.getCPCSRArrays(&r, &o, &c, &a);
  if (r.size() != 1) ++errs;
  if (r[0] != 3) ++errs;
  if (o.size() != 2) ++errs;
  if (o[0] != 0) ++errs;
  if (o[1] != 1) ++errs;
  if (c.size() != 1) ++errs;
  if (c[0] != 0) ++ errs;
  if (a.size() != 1) ++errs;
  if (a[0] != 4) ++errs;

  matrix.addEntry(3, 0, 1);
  matrix.getCPCSRArrays(&r, &o, &c, &a);
  if (r.size() != 1) ++errs;
  if (r[0] != 3) ++errs;
  if (o.size() != 2) ++errs;
  if (o[0] != 0) ++errs;
  if (o[1] != 1) ++errs;
  if (c.size() != 1) ++errs;
  if (c[0] != 0) ++ errs;
  if (a.size() != 1) ++errs;
  if (fabs(a[0] - 5) > EPS_D) ++errs;

  matrix.addEntry(2, 3, 1);
  matrix.getCPCSRArrays(&r, &o, &c, &a);
  if (r.size() != 2) ++errs;
  if (r[0] != 2) ++errs;
  if (r[1] != 3) ++errs;
  if (o.size() != 3) ++errs;
  if (o[0] != 0) ++errs;
  if (o[1] != 1) ++errs;
  if (o[2] != 2) ++errs;
  if (c.size() != 2) ++errs;
  if (c[0] != 3) ++ errs;
  if (c[1] != 0) ++ errs;
  if (a.size() != 2) ++errs;
  if (fabs(a[0] - 1) > EPS_D) ++errs;
  if (fabs(a[1] - 5) > EPS_D) ++errs;

  return errs;
}
int spmvTest() {
  int errs = 0;

  CPCSR matrix(11, 22);
  matrix.addEntry(2, 3, 11.0);
  matrix.addEntry(3, 3, 22.0);
  matrix.addEntry(2, 6, 33.0);
  matrix.addEntry(2, 3, 10.0);
  matrix.addEntry(0, 0, 44.0);

  if (fabs(computeMatrixElement(matrix, 0, 0) - 44.0) > EPS_D) ++errs;
  if (fabs(computeMatrixElement(matrix, 2, 3) - 21.0) > EPS_D) ++errs;
  if (fabs(computeMatrixElement(matrix, 3, 3) - 22.0) > EPS_D) ++errs;
  if (fabs(computeMatrixElement(matrix, 2, 6) - 33.0) > EPS_D) ++errs;

  if (fabs(computeMatrixElement(matrix, 0, 1) - 0) > EPS_D) ++errs;
  if (fabs(computeMatrixElement(matrix, 0, 2) - 0) > EPS_D) ++errs;
  if (fabs(computeMatrixElement(matrix, 0, 3) - 0) > EPS_D) ++errs;
  if (fabs(computeMatrixElement(matrix, 1, 6) - 0) > EPS_D) ++errs;
  if (fabs(computeMatrixElement(matrix, 3, 4) - 0) > EPS_D) ++errs;
  if (fabs(computeMatrixElement(matrix, 3, 2) - 0) > EPS_D) ++errs;

  return errs;
}
