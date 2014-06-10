#ifndef BACKEND_HPP
#define BACKEND_HPP

class TxMatrixOptimizationDataBase;
class TxVectorOptimizationDataBase;

class Backend {
  public:
    Backend(TxMatrixOptimizationDataBase* m, TxVectorOptimizationDataBase* v) :
      matOptData(m), vecOptData(v) {}
    TxMatrixOptimizationDataBase* getMatOptData() { return matOptData;}
    TxVectorOptimizationDataBase* getVecOptData() { return vecOptData;}
  private:
    TxMatrixOptimizationDataBase* matOptData;
    TxVectorOptimizationDataBase* vecOptData;
};

#endif
