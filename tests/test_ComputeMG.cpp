#include <config.h>
#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>
#include <boost/test/output_test_stream.hpp>
using namespace boost::unit_test;
using boost::test_tools::output_test_stream;
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <Vector.hpp>
#include <SparseMatrix.hpp>
#include <ComputeMG.hpp>
#include <ExchangeHalo.hpp>

#include <MPIFixture.hpp>
#include <HPCGFixture.hpp>
#include <testUtils.hpp>


BOOST_GLOBAL_FIXTURE(MPIFixture);

/**
 * @brief reference implementation for MG.
 *
 * taken from HPCG v2.1
 * */
int ComputeMG_ref(const SparseMatrix & A, const Vector & r, Vector & x);
int ComputeProlongation_ref(const SparseMatrix & Af, Vector & xf);
int ComputeRestriction_ref(const SparseMatrix &A, const Vector &rf);
int ComputeSPMV_ref(const SparseMatrix &A, Vector &x, Vector &y);
int ComputeSYMGS_ref(const SparseMatrix &A, const Vector &r, Vector &x);

/**
 * @brief compare ComputeMG_ref and ComputeMG
 * */
void compareMGWithRef(SparseMatrix& m);

/**
 * @brief Compare SYMGS with reference implementation.
 * */
void testComputeMG(Dims d) {
  char** options = buildOptions(d.nx, d.ny, d.nz);
  HPCG_Fixture fixture(3, options);
  destroyOptions(options);

  BOOST_TEST_CHECKPOINT("Calling compareMGWithRef in testComputeMG");
  compareMGWithRef(fixture.A);
}

void compareMGWithRef(SparseMatrix& m) {
  Vector r;
  InitializeVector(r, m.localNumberOfRows);
  FillRandomVector(r);
  Vector x_ref;
  InitializeVector(x_ref, m.localNumberOfColumns);
  FillRandomVector(x_ref);
  Vector x;
  InitializeVector(x, m.localNumberOfColumns);
  CopyVector(x_ref, x);

  int err = ComputeMG_ref(m, r, x_ref);
  BOOST_REQUIRE_EQUAL(0, err);

  err = ComputeMG(m, r, x);
  BOOST_REQUIRE_EQUAL(0, err);

  CHECK_RANGES_EQUAL(x_ref.values, x_ref.values + m.localNumberOfRows, x.values,
                     1.0e-8);

//  for (int i = 0; i < m.localNumberOfRows; ++i) {
//    std::cout << i << " " << x_ref.values[i] << " " << x.values[i] << std::endl;
//  }

  DeleteVector(r);
  DeleteVector(x);
  DeleteVector(x_ref);
}

/**
 * @brief Test driver.
 * */
test_suite* init_unit_test_suite(int argc, char *argv[])
{
  std::vector<Dims> geometries;
  geometries.push_back(Dims(16, 16, 16));
  geometries.push_back(Dims(16, 16, 32));
  geometries.push_back(Dims(32, 16, 16));
  geometries.push_back(Dims(16, 32, 16));
  geometries.push_back(Dims(16, 48, 16));
  test_suite* fine_tests = BOOST_TEST_SUITE("Fine Matrix test suite");
  fine_tests->add(BOOST_PARAM_TEST_CASE(
      &testComputeMG, geometries.begin(), geometries.end()));
  framework::master_test_suite().add(fine_tests);
  return 0;
}

int ComputeMG_ref(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return(ierr);
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr!=0) return(ierr);
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return(ierr);
    ierr = ComputeMG_ref(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return(ierr);
    ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return(ierr);
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return(ierr);
  }
  else {
    ierr = ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return(ierr);
  }
  return(0);
}

int ComputeProlongation_ref(const SparseMatrix & Af, Vector & xf) {

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

  for (local_int_t i=0; i<nc; ++i) xfv[f2c[i]] += xcv[i];

  return(0);
}

int ComputeSPMV_ref(const SparseMatrix &A, Vector &x, Vector &y)
{
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);
#ifndef HPCG_NOMPI
  ExchangeHalo(A, x);
#endif
  const double *const xv = x.values;
  double *const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
  for (local_int_t i = 0; i < nrow; i++) {
    double sum = 0.0;
    const double *const cur_vals = A.matrixValues[i];
    const local_int_t *const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j = 0; j < cur_nnz; j++)
      sum += cur_vals[j] * xv[cur_inds[j]];
    yv[i] = sum;
  }
  return (0);
}

int ComputeRestriction_ref(const SparseMatrix &A, const Vector &rf) {

  double *Axfv = A.mgData->Axf->values;
  double *rfv = rf.values;
  double *rcv = A.mgData->rc->values;
  local_int_t *f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

  for (local_int_t i = 0; i < nc; ++i)
    rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

  return (0);
}

int ComputeSYMGS_ref(const SparseMatrix &A, const Vector &r, Vector &x) {
  assert(x.localLength == A.localNumberOfColumns);
#ifndef HPCG_NOMPI
  ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal;
  const double *const rv = r.values;
  double *const xv = x.values;

  // Forward sweep
  for (local_int_t i = 0; i < nrow; i++) {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];      
    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] * currentDiagonal;
    xv[i] = sum / currentDiagonal;
  }

  // Now the back sweep.
  for (local_int_t i = nrow - 1; i >= 0; i--) {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];       
    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] * currentDiagonal;
    xv[i] = sum / currentDiagonal;
  }
  return (0);
}

