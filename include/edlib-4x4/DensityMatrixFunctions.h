#ifndef HUBBARD_ENTANGLEMENTENTROPY_H
#define HUBBARD_ENTANGLEMENTENTROPY_H

#include "edlib/DensityMatrix.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace EDLib {
  template<class Hamiltonian>
  class DensityMatrixFunctions {
  protected:
    typedef typename Hamiltonian::ModelType::precision precision;
    typedef typename Hamiltonian::ModelType::Sector sector;
    typedef typename Hamiltonian::ModelType::SYMMETRY symmetry;

  public:

    DensityMatrixFunctions(alps::params &p, DensityMatrix<Hamiltonian>& dm_) :
      dm(dm_)
    {
    }

    /**
     * Compute Tr(densitymatrix) - Tr(densitymatrix^2).
     */
    double tr_trsq(){
      double Tr = 0.0;
      double Tr_sq = 0.0;
      for(size_t isect = 0; isect < dm.sectors().size(); ++isect){
        for(size_t ii = 0; ii < dm.sectors()[isect].size(); ++ii){
          Tr += dm.matrix().at(isect)[ii][ii];
          for(size_t jj = 0; jj < dm.sectors()[isect].size(); ++jj){
            for(size_t kk = 0; kk < dm.sectors()[isect].size(); ++kk){
              Tr_sq += dm.matrix().at(isect)[jj][jj] * dm.matrix().at(isect)[kk][kk];
            }
          }
        }
      }
      return Tr_sq - Tr;
    }

    /**
     * Compute entanglement entropy.
     */
    double S_entanglement(){
      std::vector<std::vector<precision>> full = dm.full();
      Eigen::Matrix<precision, Eigen::Dynamic, Eigen::Dynamic> rho(full.size(), full.size());
      rho.fill(0.0);
      for(size_t ii = 0; ii < full.size(); ++ii){
        for(size_t jj = ii; jj < full.size(); ++jj){
          rho(ii, jj) = full[ii][jj];
        }
      }
      Eigen::Matrix<precision, Eigen::Dynamic, Eigen::Dynamic> rho_logrho = rho * rho.log();
      return -rho_logrho.trace();
    }

  private:

    DensityMatrix<Hamiltonian>& dm;

  };
}

#endif
