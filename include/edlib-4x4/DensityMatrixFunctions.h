#ifndef E4X4_DENSITYMATRIXFUNCTIONS_H
#define E4X4_DENSITYMATRIXFUNCTIONS_H

#include "ext/DensityMatrix.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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
     * Compute quadratic entanglement entropy Tr(rho - rho^2).
     */
    precision tr_trsq(){
      precision Tr = 0.0;
      precision Tr_sq = 0.0;
      for(size_t isect = 0; isect < dm.sectors().size(); ++isect){
        for(size_t ii = 0; ii < dm.sectors()[isect].size(); ++ii){
          Tr += dm.matrix().at(isect)[ii][ii];
          for(size_t jj = 0; jj < dm.sectors()[isect].size(); ++jj){
            Tr_sq += dm.matrix().at(isect)[ii][jj] * dm.matrix().at(isect)[jj][ii];
          }
        }
      }
      return Tr - Tr_sq;
    }

    /**
     * Compute Von Neumann entropy.
     */
    precision S_entanglement(){
      /* Optimised: Tr(M * log(M)) = Tr(eigenvalues(M) * log(eigenvalues(M))) */
      std::vector<precision> espec = entanglement_spectrum();
      precision sum = 0.0;
      for(size_t ii = 0; ii < espec.size(); ++ii){
        // XXX I'm not sure this is right!
        if(std::abs(espec[ii]) > 1e-9){
          sum -= espec[ii] * std::log(espec[ii]);
        }
      }
      return sum;
    }

    /**
     * Compute entanglement spectrum, i.e. eigenvalues of the density matrix.
     */
    std::vector<precision> entanglement_spectrum(){
      std::vector<precision> result(0);
      for(size_t isect = 0; isect < dm.sectors().size(); ++isect){
       Eigen::Matrix<precision, Eigen::Dynamic, Eigen::Dynamic> rho(dm.sectors()[isect].size(), dm.sectors()[isect].size());
       for(size_t ii = 0; ii < dm.sectors()[isect].size(); ++ii){
         for(size_t jj = 0; jj < dm.sectors()[isect].size(); ++jj){
           rho(ii, jj) = dm.matrix().at(isect)[ii][jj];
         }
       }
       // FIXME This won't compile with "expected expression". Whatever gets the job done...
       //Eigen::Matrix<precision, Eigen::Dynamic, 1> evals = rho.selfadjointView<Eigen::Lower>().eigenvalues();
       Eigen::Matrix<std::complex<precision>, Eigen::Dynamic, 1> evals = rho.eigenvalues();
       for(size_t ii = 0; ii < evals.size(); ++ii){
        result.push_back(evals(ii).real());
       }
      }
      std::sort(result.begin(), result.end());
      return result;
    }

  private:

    DensityMatrix<Hamiltonian>& dm;

  };
}

#endif
