#ifndef CPCSR_H
#define CPCSR_H

#include <vector>
#include <map>
#include <cstddef>

#include <Geometry.hpp>

/**
 * Class representing a column index-value pair.
 **/
struct ColValPair {
  ColValPair() : col(0), val(0) {}
  ColValPair(local_int_t j, double a) : col(j), val(a) {}
  local_int_t col;
  double val;
};

/**
 * @brief Class for very sparse matrices.
 *
 * This is a simplified version of a very sparse matrix.  It can be
 * used, e.g. to scatter from the halo into the local domain in hpcg.
 * Only non-zero rows are stored.
 *
 * This matrix is not suitable for very large problems (many non-zeros).
 **/
class CPCSR {
 public:
  CPCSR(local_int_t nRows = 0, local_int_t nCols = 0);
  CPCSR(const CPCSR& other);
  CPCSR& operator=(const CPCSR& rhs);
  ~CPCSR();

  /** 
   * @brief Adds a non-zero entry to the matrix. 
   *
   * If the matrix already has an entry at location (i, j) the val is
   * being added to the existing entry.  Otherwise an entry is created.
   *
   * @param i row index
   * @param j column index
   * @param val value
   **/
  void addEntry(local_int_t i, local_int_t j, double val);

  /**
   * @brief Compute y <- alpha * A * x + beta * y 
   *
   * @param x device pointer to x
   * @param y device pointer to y
   **/
  void spmv(double* x, double* y, double alpha = 1, double beta = 1);

  /**
   * @brief Returns the total number of non-zeros in the matrix
   **/
  size_t getNNZ() const;

  /**
   * @brief Returns the number of rows with a non-zero entry
   **/
  size_t getNumNonZeroRows() const;

  /**
   * @brief Returns arrays representing the matrix in CPCSR format
   *
   * The arrays are resized (to the lengths indicated below) and their content
   * is overwritten.
   *
   * @param r [out] Nonzero rows (of length getNumNonZeroRows())
   * @param o [out] Offsets for non-zero rows into c and a arrays (of
   *                length getNumNonZeroRows() + 1).  This corresponds
   *                to the i array in csr, except that it only contains
   *                the offsets for non-zero rows.
   * @param c [out] Column indices (of length getNNZ()).  This
   *                corresponds to the j array in csr.
   * @param a [out] Values of non-zero entries(of length getNNZ().  This
   *                corresponds to the a array in csr.
   **/
  void getCPCSRArrays(std::vector<local_int_t>* r, std::vector<local_int_t>* o,
                      std::vector<local_int_t>* c,
                      std::vector<double>* a) const;

  /**
   * @brief Set the number of rows in the matrix.
   **/
  void setNumRows(local_int_t m) { numRows = m; }

  /**
   * @brief Get the number of rows in the matrix.
   **/
  local_int_t getNumRows() const { return numRows; }

  /**
   * @brief Set the number of columns
   **/
  void setNumCols(local_int_t n) { numCols = n; }

  /**
   * @brief Get the number of columns in the matrix.
   **/
  local_int_t getNumCols() const { return numCols; }

  /**
   * @brief Remove all entries from the matrix
   **/
  void clear();

 private:
  local_int_t numRows;
  local_int_t numCols;
  std::map<local_int_t, std::vector<ColValPair> > entries;

  // GPU data
  bool GPUDataValid;
  local_int_t* nonZeroRows;
  local_int_t* offsets;
  local_int_t* columnIndices;
  double* values;

  // Delete device array.
  void delDevVec(void* ptr);
  // Copy data between two device vectors.
  // dest is resized to size. The dest pointer invalidated.
  void deviceCopy(void** dest, const void* src, size_t size);

  // Serialize the matrix data into flat arrays and transfer to GPU.
  void buildGPUData();
};

void merge(std::vector<ColValPair>* v, const std::vector<ColValPair>& w);
std::vector<ColValPair>::iterator findDup(const ColValPair& ja,
                                          std::vector<ColValPair>& v);

#endif
