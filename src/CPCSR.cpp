#include "CPCSR.hpp"
#include "chkcudaerror.hpp"
#include "KernelWrappers.h"

CPCSR::CPCSR(local_int_t nRows, local_int_t nCols)
    : numRows(nRows),
      numCols(nCols),
      GPUDataValid(false),
      nonZeroRows(0),
      offsets(0),
      columnIndices(0),
      values(0) {}

CPCSR::CPCSR(const CPCSR& other)
    : numRows(other.numRows),
      numCols(other.numCols),
      entries(other.entries),
      GPUDataValid(other.GPUDataValid) {
  if (GPUDataValid) {
    CHKCUDAERR(cudaMemcpy(nonZeroRows, other.nonZeroRows,
                          entries.size() * sizeof(local_int_t),
                          cudaMemcpyDeviceToDevice));
    CHKCUDAERR(cudaMemcpy(offsets, other.offsets,
                          (entries.size() + 1) * sizeof(local_int_t),
                          cudaMemcpyDeviceToDevice));
    size_t nnz = getNNZ();
    CHKCUDAERR(cudaMemcpy(columnIndices, other.columnIndices,
                          nnz * sizeof(local_int_t), cudaMemcpyDeviceToDevice));
    CHKCUDAERR(cudaMemcpy(values, other.values, nnz * sizeof(local_int_t),
                          cudaMemcpyDeviceToDevice));
  } else {
    nonZeroRows = 0;
    offsets = 0;
    columnIndices = 0;
    values = 0;
  }
}

CPCSR& CPCSR::operator=(const CPCSR& rhs) {
  if (this != &rhs) {
    numRows = rhs.numRows;
    numCols = rhs.numCols;
    entries = rhs.entries;
    GPUDataValid = rhs.GPUDataValid;
    if (GPUDataValid) {
      deviceCopy((void**)&nonZeroRows, rhs.nonZeroRows,
                 entries.size() * sizeof(local_int_t));
      deviceCopy((void**)&offsets, rhs.offsets,
                 (entries.size() + 1) * sizeof(local_int_t));
      size_t nnz = getNNZ();
      deviceCopy((void**)&columnIndices, rhs.columnIndices,
                 nnz * sizeof(local_int_t));
      deviceCopy((void**)&values, rhs.values, nnz * sizeof(double));
    } else {
      nonZeroRows = 0;
      offsets = 0;
      columnIndices = 0;
      values = 0;
    }
  }
  return *this;
}

CPCSR::~CPCSR() {
  delDevVec(nonZeroRows);
  delDevVec(offsets);
  delDevVec(columnIndices);
  delDevVec(values);
}

void CPCSR::addEntry(local_int_t i, local_int_t j, double val) {
  if (i < 0 || i > numRows) {
    throw std::runtime_error("Invalid row index in addEntry.");
  }
  if (j < 0 || j > numCols) {
    throw std::runtime_error("Invalid column index in addEntry.");
  }
  GPUDataValid = false;
  ColValPair ja;
  ja.col = j;
  ja.val = val;
  std::vector<ColValPair> newRow(1, ja);
  std::pair<std::map<local_int_t, std::vector<ColValPair> >::iterator, bool>
      ret = entries.insert(
          std::pair<local_int_t, std::vector<ColValPair> >(i, newRow));
  if (ret.second == false) {
    merge(&ret.first->second, newRow);
  }
}

void CPCSR::spmv(double* x, double* y, double alpha, double beta) {
  if (!GPUDataValid) {
    buildGPUData();
    GPUDataValid = true;
  }
  launchSpmvKernel(getNumNonZeroRows(), nonZeroRows, offsets, columnIndices,
                   values, x, y, alpha, beta);
}

void CPCSR::buildGPUData() {
  std::vector<local_int_t> r;
  std::vector<local_int_t> o;
  std::vector<local_int_t> c;
  std::vector<double> a;
  getCPCSRArrays(&r, &o, &c, &a);

  delDevVec(nonZeroRows);
  CHKCUDAERR(cudaMalloc((void**)&nonZeroRows, sizeof(local_int_t) * r.size()));
  CHKCUDAERR(cudaMemcpy(nonZeroRows, &r[0], sizeof(local_int_t) * r.size(),
                        cudaMemcpyHostToDevice));
  delDevVec(offsets);
  CHKCUDAERR(cudaMalloc((void**)&offsets, sizeof(local_int_t) * o.size()));
  CHKCUDAERR(cudaMemcpy(offsets, &o[0], sizeof(local_int_t) * o.size(),
                        cudaMemcpyHostToDevice));
  delDevVec(columnIndices);
  CHKCUDAERR(
      cudaMalloc((void**)&columnIndices, sizeof(local_int_t) * c.size()));
  CHKCUDAERR(cudaMemcpy(columnIndices, &c[0], sizeof(local_int_t) * c.size(),
                        cudaMemcpyHostToDevice));
  delDevVec(values);
  CHKCUDAERR(cudaMalloc((void**)&values, sizeof(double) * a.size()));
  CHKCUDAERR(cudaMemcpy(values, &a[0], sizeof(double) * a.size(),
                        cudaMemcpyHostToDevice));
}

size_t CPCSR::getNNZ() const {
  size_t nnz = 0;
  for (std::map<local_int_t, std::vector<ColValPair> >::const_iterator i =
           entries.begin();
       i != entries.end(); ++i) {
    nnz += i->second.size();
  }
  return nnz;
}

size_t CPCSR::getNumNonZeroRows() const { return entries.size(); }

void CPCSR::getCPCSRArrays(std::vector<local_int_t>* r,
                           std::vector<local_int_t>* o,
                           std::vector<local_int_t>* c,
                           std::vector<double>* a) const {
  r->resize(entries.size());
  o->resize(entries.size() + 1);
  o->at(0) = 0;
  c->resize(getNNZ());
  a->resize(getNNZ());
  size_t entry = 0;
  size_t row = 0;
  for (std::map<local_int_t, std::vector<ColValPair> >::const_iterator i =
           entries.begin();
       i != entries.end(); ++i) {
    r->at(row) = i->first;
    const std::vector<ColValPair>& rowData = i->second;
    o->at(row + 1) = o->at(row) + rowData.size();
    for (std::vector<ColValPair>::const_iterator j = rowData.begin();
        j != rowData.end(); ++j) {
      c->at(entry) = j->col;
      a->at(entry) = j->val;
      ++entry;
    }
    ++row;
  }
}

void merge(std::vector<ColValPair>* v, const std::vector<ColValPair>& w) {
  for (std::vector<ColValPair>::const_iterator i = w.begin(); i != w.end();
       ++i) {
    std::vector<ColValPair>::iterator duplicate = findDup(*i, *v);
    if (duplicate == v->end()) {
      v->push_back(*i);
    } else {
      duplicate->val += i->val;
    }
  }
}

std::vector<ColValPair>::iterator findDup(const ColValPair& ja,
                                          std::vector<ColValPair>& v) {
  std::vector<ColValPair>::iterator result = v.begin();
  while (result != v.end() && result->col != ja.col) {
    ++result;
  }
  return result;
}

void CPCSR::delDevVec(void* ptr) {
  if (ptr) {
    CHKCUDAERR(cudaFree(ptr));
    ptr = 0;
  }
}

void CPCSR::deviceCopy(void** dest, const void* src, size_t size) {
  delDevVec(*dest);
  CHKCUDAERR(cudaMalloc(dest, size));
  CHKCUDAERR(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice));
}

void CPCSR::clear() {
  entries.clear();
  delDevVec(nonZeroRows);
  delDevVec(offsets);
  delDevVec(columnIndices);
  delDevVec(values);
  GPUDataValid = false;
}
