#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <SparseMatrix.hpp>

/**
 * @brief Construction of minimal sparse matrix for unit  tests.
 * */
SparseMatrix buildSparseMatrix(int nXFine, int nYFine, int nZFine);

/**
 * @brief Functions for required floating point accuracy.
 *
 * In percent, i.e. 1.0e-5 corresponds to an error of 1.0e-7.
 * */
inline double EPS(float) { return 1.0e-5; }
inline double EPS(double) { return 1.0e-12; }

/**
 * @brief Check if two ranges are equal.
 *
 * Uses EPS() for deciding floating point number closeness.
 *
 * @sa EPS
 * */
template <typename Iter>
void CHECK_RANGES_EQUAL(Iter beginA, Iter endA, Iter beginB, double eps = 0) {
  if (eps == 0) {
    eps = EPS(*beginA);
  }
  size_t i = 0;
  for (; beginA != endA; ++beginA, ++beginB, ++i) {
    BOOST_CHECK_CLOSE(*beginA, *beginB, eps);
  }
}

/**
 * @brief Geometry of domain.
 * */
struct Dims {
  Dims(int x, int y, int z) : nx(x), ny(y), nz(z) {}
  int nx;
  int ny;
  int nz;
};

/**
 * @brief Build options string specifying HPCG geometry
 *
 * Suitable for passing to HPCG_Init.  The options strings should be
 * deallocated with destroyOptions.
 *
 * @return Array of size 4 with strings for options.
 *
 * @sa destroyOptions
 * */
char** buildOptions(int nx, int ny, int nz);

/**
 * @brief Deallocate options strings
 *
 * It is assumed that the options were allocated using buildOptions.
 *
 * @sa buildOptions
 * */
void destroyOptions(char** options);

void GenerateGeometry(int size, int rank, int numThreads, int nx, int ny,
                      int nz, Geometry *geom);

#endif
