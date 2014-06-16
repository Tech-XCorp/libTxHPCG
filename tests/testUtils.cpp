#include "testUtils.hpp"
#include <SetupHalo.hpp>
#include <config.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include <cstring>
#include <cmath>
#include <sstream>


void GenerateProblem(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact) {

  // Make local copies of geometry information.  Use global_int_t since the RHS
  // products in the calculations
  // below may result in global range values.
  global_int_t nx = A.geom->nx;
  global_int_t ny = A.geom->ny;
  global_int_t nz = A.geom->nz;
  global_int_t npx = A.geom->npx;
  global_int_t npy = A.geom->npy;
  global_int_t npz = A.geom->npz;
  global_int_t ipx = A.geom->ipx;
  global_int_t ipy = A.geom->ipy;
  global_int_t ipz = A.geom->ipz;
  global_int_t gnx = nx * npx;
  global_int_t gny = ny * npy;
  global_int_t gnz = nz * npz;

  local_int_t localNumberOfRows =
      nx * ny * nz; // This is the size of our subblock
  // If this assert fails, it most likely means that the local_int_t is set to
  // int and should be set to long long
  assert(localNumberOfRows > 0); // Throw an exception of the number of rows is
                                 // less than zero (can happen if int overflow)
  local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point
                                           // finite element/volume/difference
                                           // 3D stencil

  global_int_t totalNumberOfRows =
      localNumberOfRows * A.geom->size; // Total number of grid points in mesh
  // If this assert fails, it most likely means that the global_int_t is set to
  // int and should be set to long long
  assert(totalNumberOfRows > 0); // Throw an exception of the number of rows is
                                 // less than zero (can happen if int overflow)

  // Allocate arrays that are of length localNumberOfRows
  char *nonzerosInRow = new char[localNumberOfRows];
  global_int_t **mtxIndG = new global_int_t *[localNumberOfRows];
  local_int_t **mtxIndL = new local_int_t *[localNumberOfRows];
  double **matrixValues = new double *[localNumberOfRows];
  double **matrixDiagonal = new double *[localNumberOfRows];

  if (b != 0)
    InitializeVector(*b, localNumberOfRows);
  if (x != 0)
    InitializeVector(*x, localNumberOfRows);
  if (xexact != 0)
    InitializeVector(*xexact, localNumberOfRows);
  double *bv = 0;
  double *xv = 0;
  double *xexactv = 0;
  if (b != 0)
    bv = b->values; // Only compute exact solution if requested
  if (x != 0)
    xv = x->values; // Only compute exact solution if requested
  if (xexact != 0)
    xexactv = xexact->values; // Only compute exact solution if requested
  A.localToGlobalMap.resize(localNumberOfRows);

  for (local_int_t i = 0; i < localNumberOfRows; ++i) {
    matrixValues[i] = 0;
    matrixDiagonal[i] = 0;
    mtxIndG[i] = 0;
    mtxIndL[i] = 0;
  }
  for (local_int_t i = 0; i < localNumberOfRows; ++i) {
    mtxIndL[i] = new local_int_t[numberOfNonzerosPerRow];
    matrixValues[i] = new double[numberOfNonzerosPerRow];
    mtxIndG[i] = new global_int_t[numberOfNonzerosPerRow];
  }

  local_int_t localNumberOfNonzeros = 0;
  for (local_int_t iz = 0; iz < nz; iz++) {
    global_int_t giz = ipz * nz + iz;
    for (local_int_t iy = 0; iy < ny; iy++) {
      global_int_t giy = ipy * ny + iy;
      for (local_int_t ix = 0; ix < nx; ix++) {
        global_int_t gix = ipx * nx + ix;
        local_int_t currentLocalRow = iz * nx * ny + iy * nx + ix;
        global_int_t currentGlobalRow = giz * gnx * gny + giy * gnx + gix;
        A.globalToLocalMap[currentGlobalRow] = currentLocalRow;
        A.localToGlobalMap[currentLocalRow] = currentGlobalRow;
        char numberOfNonzerosInRow = 0;
        double *currentValuePointer = matrixValues
            [currentLocalRow]; // Pointer to current value in current row
        global_int_t *currentIndexPointerG =
            mtxIndG[currentLocalRow]; // Pointer to current index in current row
        for (int sz = -1; sz <= 1; sz++) {
          if (giz + sz > -1 && giz + sz < gnz) {
            for (int sy = -1; sy <= 1; sy++) {
              if (giy + sy > -1 && giy + sy < gny) {
                for (int sx = -1; sx <= 1; sx++) {
                  if (gix + sx > -1 && gix + sx < gnx) {
                    global_int_t curcol =
                        currentGlobalRow + sz * gnx * gny + sy * gnx + sx;
                    if (curcol == currentGlobalRow) {
                      matrixDiagonal[currentLocalRow] = currentValuePointer;
                      *currentValuePointer++ = 26.0;
                    } else {
                      *currentValuePointer++ = -1.0;
                    }
                    *currentIndexPointerG++ = curcol;
                    numberOfNonzerosInRow++;
                  } // end x bounds test
                }   // end sx loop
              }     // end y bounds test
            }       // end sy loop
          }         // end z bounds test
        }           // end sz loop
        nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
        localNumberOfNonzeros +=
            numberOfNonzerosInRow; // Protect this with an atomic
        if (b != 0)
          bv[currentLocalRow] = 26.0 - ((double)(numberOfNonzerosInRow - 1));
        if (x != 0)
          xv[currentLocalRow] = 0.0;
        if (xexact != 0)
          xexactv[currentLocalRow] = 1.0;
      } // end ix loop
    }   // end iy loop
  }     // end iz loop

  global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NOMPI
// Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
  MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
#else
  long long lnnz = localNumberOfNonzeros,
            gnnz = 0; // convert to 64 bit for MPI call
  MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
  totalNumberOfNonzeros = gnnz; // Copy back
#endif
#else
  totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
  // If this assert fails, it most likely means that the global_int_t is set to
  // int and should be set to long long
  // This assert is usually the first to fail as problem size increases beyond
  // the 32-bit integer range.
  assert(totalNumberOfNonzeros > 0); // Throw an exception of the number of
                                     // nonzeros is less than zero (can happen
                                     // if int overflow)

  A.title = 0;
  A.totalNumberOfRows = totalNumberOfRows;
  A.totalNumberOfNonzeros = totalNumberOfNonzeros;
  A.localNumberOfRows = localNumberOfRows;
  A.localNumberOfColumns = localNumberOfRows;
  A.localNumberOfNonzeros = localNumberOfNonzeros;
  A.nonzerosInRow = nonzerosInRow;
  A.mtxIndG = mtxIndG;
  A.mtxIndL = mtxIndL;
  A.matrixValues = matrixValues;
  A.matrixDiagonal = matrixDiagonal;

  return;
}

typedef struct Counter_s {
  int length; //!< number of prime factor counts (cannot exceed 32 for a 32-bit
              //integer)
  int max_counts[32 + 1]; //!< maximum value for prime factor counts
  int cur_counts[32 + 1]; //!< current prime factor counts
} Counter;

static void Counter_new(Counter *this_, int *counts, int length) {
  int i;

  this_->length = length;

  for (i = 0; i < 32; ++i) {
    this_->max_counts[i] = counts[i];
    this_->cur_counts[i] = 0;
  }
  /* terminate with 0's */
  this_->max_counts[i] = this_->cur_counts[i] = 0;
  this_->max_counts[length] = this_->cur_counts[length] = 0;
}

static void Counter_next(Counter *this_) {
  int i;

  for (i = 0; i < this_->length; ++i) {
    this_->cur_counts[i]++;
    if (this_->cur_counts[i] > this_->max_counts[i]) {
      this_->cur_counts[i] = 0;
      continue;
    }
    break;
  }
}

static int Counter_is_zero(Counter *this_) {
  int i;
  for (i = 0; i < this_->length; ++i)
    if (this_->cur_counts[i])
      return 0;
  return 1;
}

static int Counter_product(Counter *this_, int *multipliers) {
  int i, j, k = 0, x = 1;

  for (i = 0; i < this_->length; ++i)
    for (j = 0; j < this_->cur_counts[i]; ++j) {
      k = 1;
      x *= multipliers[i];
    }

  return x * k;
}

static void Counter_max_cur_sub(Counter *this_, Counter *that, Counter *res) {
  int i;

  res->length = this_->length;
  for (i = 0; i < this_->length; ++i) {
    res->max_counts[i] = this_->max_counts[i] - that->cur_counts[i];
    res->cur_counts[i] = 0;
  }
}

static void primefactor_i(int x, int *factors) {
  int i, d, sq = (int)(sqrt((double)x)) + 1L;
  div_t r;

  /* remove 2 as a factor with shifts */
  for (i = 0; x > 1 && (x & 1) == 0; x >>= 1) {
    factors[i++] = 2;
  }

  /* keep removing subsequent odd numbers */
  for (d = 3; d <= sq; d += 2) {
    while (1) {
      r = div(x, d);
      if (r.rem == 0) {
        factors[i++] = d;
        x = r.quot;
        continue;
      }
      break;
    }
  }
  if (x > 1 || i == 0) /* left with a prime or x==1 */
    factors[i++] = x;

  factors[i] = 0; /* terminate with 0 */
}
static void gen_min_area3(int n, int *f1, int *f2, int *f3) {
  int i, j, df_cnt;
  int tf1, tf2, tf3;
  int factors[32 + 1], distinct_factors[32 + 1], count_factors[32 + 1];
  Counter c_main, c1, c2;

  /* at the beginning, minimum area is the maximum area */
  double area, min_area = 2.0 * n + 1.0;

  primefactor_i(n, factors); /* factors are sorted: ascending order */

  if (1 == n || factors[1] == 0) { /* prime number */
    *f1 = n;
    *f2 = 1;
    *f3 = 1;
    return;
  } else if (factors[2] == 0) { /* two prime factors */
    *f1 = factors[0];
    *f2 = factors[1];
    *f3 = 1;
    return;
  } else if (factors[3] == 0) { /* three prime factors */
    *f1 = factors[0];
    *f2 = factors[1];
    *f3 = factors[2];
    return;
  }

  /* we have more than 3 prime factors so we need to try all possible
   * combinations */

  for (j = 0, i = 0; factors[i];) {
    distinct_factors[j++] = factors[i];
    count_factors[j - 1] = 0;
    do {
      count_factors[j - 1]++;
    } while (distinct_factors[j - 1] == factors[++i]);
  }
  df_cnt = j;

  Counter_new(&c_main, count_factors, df_cnt);

  Counter_new(&c1, count_factors, df_cnt);

  for (Counter_next(&c1); !Counter_is_zero(&c1); Counter_next(&c1)) {

    Counter_max_cur_sub(&c_main, &c1, &c2);
    for (Counter_next(&c2); !Counter_is_zero(&c2); Counter_next(&c2)) {
      tf1 = Counter_product(&c1, distinct_factors);
      tf2 = Counter_product(&c2, distinct_factors);
      tf3 = n / tf1 / tf2;

      area = tf1 * (double)tf2 + tf2 * (double)tf3 + tf1 * (double)tf3;
      if (area < min_area) {
        min_area = area;
        *f1 = tf1;
        *f2 = tf2;
        *f3 = tf3;
      }
    }
  }
}

/*!
  Computes the factorization of the total number of processes into a
  3-dimensional process grid that is as close as possible to a cube. The
  quality of the factorization depends on the prime number structure of the
  total number of processes. It then stores this decompostion together with the
  parallel parameters of the run in the geometry data structure.

  @param[in]  size total number of MPI processes
  @param[in]  rank this process' rank among other MPI processes
  @param[in]  numThreads number of OpenMP threads in this process
  @param[in]  nx, ny, nz number of grid points for each local block in the x, y,
  and z dimensions, respectively
  @param[out] geom data structure that will store the above parameters and the
  factoring of total number of processes into three dimensions
*/
void GenerateGeometry(int size, int rank, int numThreads, int nx, int ny,
                      int nz, Geometry *geom) {

  int npx, npy, npz;

  gen_min_area3(size, &npx, &npy, &npz);

  // Now compute this process's indices in the 3D cube
  int ipz = rank / (npx * npy);
  int ipy = (rank - ipz * npx * npy) / npx;
  int ipx = rank % npx;

#ifdef HPCG_DEBUG
  if (rank == 0)
    HPCG_fout << "size = " << size << endl << "nx  = " << nx << endl
              << "ny  = " << ny << endl << "nz  = " << nz << endl
              << "npx = " << npx << endl << "npy = " << npy << endl
              << "npz = " << npz << endl;

  HPCG_fout << "For rank = " << rank << endl << "ipx = " << ipx << endl
            << "ipy = " << ipy << endl << "ipz = " << ipz << endl;

  assert(size == npx * npy * npz);
#endif
  geom->size = size;
  geom->rank = rank;
  geom->numThreads = numThreads;
  geom->nx = nx;
  geom->ny = ny;
  geom->nz = nz;
  geom->npx = npx;
  geom->npy = npy;
  geom->npz = npz;
  geom->ipx = ipx;
  geom->ipy = ipy;
  geom->ipz = ipz;
  return;
}

char** buildOptions(int nx, int ny, int nz)
{
  char** options = new char*[4];
  std::stringstream opt;
  opt.str("");
  opt.clear();
  opt << "test_ComputeSPMV";
  options[0] = new char[strlen(opt.str().c_str()) + 1];
  strcpy(options[0], opt.str().c_str());
  opt.str("");
  opt.clear();
  opt << "--nx=" << nx;
  options[1] = new char[strlen(opt.str().c_str()) + 1];
  strcpy(options[1], opt.str().c_str());
  opt.str("");
  opt.clear();
  opt << "--ny=" << ny;
  options[2] = new char[strlen(opt.str().c_str()) + 1];
  strcpy(options[2], opt.str().c_str());
  opt.str("");
  opt.clear();
  opt << "--nz=" << nz;
  options[3] = new char[strlen(opt.str().c_str()) + 1];
  strcpy(options[3], opt.str().c_str());
  return options;
}

void destroyOptions(char** options)
{
  delete [] options[0];
  delete [] options[1];
  delete [] options[2];
  delete [] options[3];
  delete [] options;
}
