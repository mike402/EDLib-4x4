#ifndef E4X4_DENSITYMATRIX_H
#define E4X4_DENSITYMATRIX_H

#include "edlib/EigenPair.h"

#include <alps/params.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace EDLib {
  /**
   * Class for evaluation of the density matrix.
   *
   *
   *
   * @tparam Hamiltonian - type of Hamiltonian object
   */
  template<class Hamiltonian>
  class DensityMatrix {
  protected:
    typedef typename Hamiltonian::ModelType::precision precision;
    typedef typename Hamiltonian::ModelType::Sector sector;
    typedef typename Hamiltonian::ModelType::SYMMETRY symmetry;

  public:

    /**
     * Construct an object of the density matrix class
     *
     * @param p - AlpsCore parameter object
     * @param h - Hamiltonain instance
     * @param orbitals - the orbitals to calculate the density matrix for
     */
    DensityMatrix(alps::params &p, Hamiltonian& _ham_, std::vector<size_t> orbitals) :
      _Ns(int(p["NSITES"])),
      _Ip(int(p["NSPINS"]) * int(p["NSITES"])),
      _Nspins(int(p["NSPINS"])),
      _beta(p["lanc.BETA"].as<precision>()),
      _cutoff(p["lanc.BOLTZMANN_CUTOFF"]),
      _ham(_ham_),
      _orbsA(orbitals)
    {
      if(p["storage.EIGENVALUES_ONLY"] == 0){
        std::sort(_orbsA.begin(), _orbsA.end());
        _orbsA.erase(std::unique(_orbsA.begin(), _orbsA.end()), _orbsA.end());
        _symA = std::vector<symmetry>(2, symmetry(_orbsA.size()));
        _symB = std::vector<symmetry>(1, symmetry(_Ns - _orbsA.size()));
        _Ns_A = _orbsA.size();
        _Ns_B = _Ns - _Ns_A;
        for(int iorb = 0; iorb < _Ns; ++iorb){
          _orbsB.push_back(iorb);
        }
        for(int iorb = _orbsA.size() - 1; iorb >= 0; --iorb){
          _orbsB.erase(_orbsB.begin() + _orbsA[iorb]);
        }
        for (int i = 0; i <= _Ns_A; ++i) {
          for (int j = 0; j <= _Ns_A; ++j) {
            _secA.push_back(sector(i, j, (size_t)(
              _symA[0].comb().c_n_k(_Ns_A, i) *
              _symA[0].comb().c_n_k(_Ns_A, j)
            )));
          }
        }
        for(size_t isect = 0; isect < _secA.size(); ++isect){
          _rho.insert(
            std::pair<size_t, std::vector<std::vector<precision>>>(
              isect,
              std::vector<std::vector<precision>>(
                _secA[isect].size(),
                std::vector<precision>(_secA[isect].size(), 0.0)
              )
            )
          );
        }
      }else{
       _Ns_A = 0;
       std::cout << "Density matrix can not be calculated. " << std::endl;
      }
      invalidate_cache();
    }

    /**
     * Compute reduced density matrix.
     *
     * @return sectors of the reduced density matrix
     */
    const std::map<size_t, std::vector<std::vector<precision>>> &compute() {
      invalidate_cache();
      for(size_t isect = 0; isect < _secA.size(); ++isect){
        for(size_t jj = 0; jj < _secA[isect].size(); ++jj){
          for(size_t kk = 0; kk < _secA[isect].size(); ++kk){
           _rho[isect][jj][kk] = 0.0;
          }
        }
      }
      precision sum = 0.0;
      const EigenPair<precision, sector> &groundstate = *_ham.eigenpairs().begin();
      // Loop over all eigenpairs.
      for(auto ipair = _ham.eigenpairs().begin(); ipair != _ham.eigenpairs().end(); ++ipair){
        const EigenPair<precision, sector>& pair = *ipair;
        // Calculate Boltzmann factor, skip the states with trivial contribution.
        precision boltzmann_f = std::exp(
         -(pair.eigenvalue() - groundstate.eigenvalue()) * _beta
        );
        if(boltzmann_f < _cutoff){
          continue;
        }
        // Sum the contributions.
        compute_eigenvector(pair, boltzmann_f);
        sum += boltzmann_f;
      }
      for(size_t isect = 0; isect < _secA.size(); ++isect){
        for(size_t jj = 0; jj < _secA[isect].size(); ++jj){
          for(size_t kk = 0; kk < _secA[isect].size(); ++kk){
           _rho[isect][jj][kk] /= sum;
          }
        }
      }
      return _rho;
    }

    /**
     * Print the reduced density matrix.
     */
    void print() {
#ifdef USE_MPI
      int myid;
      MPI_Comm_rank(_ham.comm(), &myid);
      if(!myid)
#endif
      for(size_t isect = 0; isect < _secA.size(); ++isect){
        std::cout << "Density matrix sector " << _secA[isect].nup() << " " << _secA[isect].ndown() << std::endl;
        for(size_t jj = 0; jj < _secA[isect].size(); ++jj){
          for(size_t kk = 0; kk < _secA[isect].size(); ++kk){
            if(kk){
              std::cout << "\t";
            }
            std::cout << _rho[isect][jj][kk];
          }
          std::cout << std::endl;
        }
      }
    }

    /**
     * Combine the sectors of reduced density matrix.
     *
     * @return the whole reduced density matrix
     */
    const std::vector<std::vector<precision>> full() const {
      size_t fullsize = std::pow(2, _Ns_A * _Nspins);
      std::vector<std::vector<precision>> result(
        fullsize, std::vector<precision>(fullsize, 0.0)
      );
      size_t shift = 0;
      for(size_t isect = 0; isect < _secA.size(); ++isect){
        for(size_t ii = 0; ii < _secA[isect].size(); ++ii){
          for(size_t jj = 0; jj < _secA[isect].size(); ++jj){
            result[shift + ii][shift + jj] = _rho.at(isect)[ii][jj];
          }
        }
        shift += _secA[isect].size();
      }
      return result;
    }

    /**
     * Compute quadratic entanglement entropy Tr(rho - rho^2).
     */
    precision quadratic_entropy(){
      if(!_quadratic_entropy_valid){
        _quadratic_entropy= 0.0;
        for(size_t isect = 0; isect < _secA.size(); ++isect){
          for(size_t ii = 0; ii < _secA[isect].size(); ++ii){
            _quadratic_entropy += _rho[isect][ii][ii];
            for(size_t jj = 0; jj < _secA[isect].size(); ++jj){
              _quadratic_entropy -= _rho[isect][ii][jj] * _rho[isect][jj][ii];
            }
          }
        }
      }
      return _quadratic_entropy;
    }

    /**
     * Compute Von Neumann entropy -Tr(rho * log(rho)).
     */
    precision entanglement_entropy(){
      if(!_entanglement_entropy_valid){
        _entanglement_entropy = 0.0;
        eigenvalues();
        /* Optimised: Tr(M * log(M)) = Tr(eigenvalues(M) * log(eigenvalues(M))) */
        for(size_t ii = 0; ii < _eigenvalues.size(); ++ii){
          // XXX I'm not sure this is right!
          if(std::abs(_eigenvalues[ii]) > 1e-9){
            _entanglement_entropy -= _eigenvalues[ii] * std::log(_eigenvalues[ii]);
          }
        }
        _entanglement_entropy_valid = true;
      }
      return _entanglement_entropy;
    }

    /**
     * Compute entanglement spectrum, i.e. eigenvalues of the density matrix.
     */
    std::vector<precision> eigenvalues(){
      if(!_eigenvalues_valid){
        _eigenvalues.clear();
        for(size_t isect = 0; isect < _secA.size(); ++isect){
         Eigen::Matrix<precision, Eigen::Dynamic, Eigen::Dynamic> M(_secA[isect].size(), _secA[isect].size());
         for(size_t ii = 0; ii < _secA[isect].size(); ++ii){
           for(size_t jj = 0; jj < _secA[isect].size(); ++jj){
             M(ii, jj) = _rho[isect][ii][jj];
           }
         }
         Eigen::Matrix<std::complex<precision>, Eigen::Dynamic, 1> evals = M.eigenvalues();
         for(size_t ii = 0; ii < evals.size(); ++ii){
          _eigenvalues.push_back(evals(ii).real());
         }
        }
        std::sort(_eigenvalues.begin(), _eigenvalues.end(), [](precision a, precision b) {return a > b;});
        _eigenvalues_valid = true;
      }
      return _eigenvalues;
    }

  inline const std::vector<sector> &sectors()
  const {
   return _secA;
  }

  inline const std::map<size_t, std::vector<std::vector<precision>>> &matrix()
  const {
   return _rho;
  }

  inline const Hamiltonian &ham()
  const {
   return _ham;
  }

  private:

    /**
     * Compute the contribution of an eigenvector to the density matrix.
     *
     * @param pair   the eigenpair
     * @param weight additional multiplier (Boltzmann factor)
     */
    void compute_eigenvector(const EigenPair<precision, sector>& pair, precision weight) {
#ifdef USE_MPI
      int myid;
      int nprocs;
      MPI_Comm_rank(_ham.comm(), &myid);
      MPI_Comm_size(_ham.comm(), &nprocs);
      std::vector<int> counts(nprocs);
      std::vector<int> displs(nprocs + 1);
      std::vector<precision> evec(pair.sector().size());
      int size = pair.eigenvector().size();
      MPI_Allgather(&size, 1,
                    alps::mpi::detail::mpi_type<int>(),
                    counts.data(), 1,
                    alps::mpi::detail::mpi_type<int>(),
                    _ham.comm()
      );
      displs[0] = 0;
      for(size_t i = 0; i < nprocs; ++i){
       displs[i + 1] = displs[i] + counts[i];
      }
      MPI_Allgatherv(const_cast<precision *>(pair.eigenvector().data()), pair.eigenvector().size(),
                     alps::mpi::detail::mpi_type<precision>(),
                     evec.data(), counts.data(), displs.data(),
                     alps::mpi::detail::mpi_type<precision>(),
                     _ham.comm()
      );
#endif
      _ham.model().symmetry().set_sector(pair.sector());
      for(size_t isect = 0; isect < _secA.size(); ++isect){
        int nupB = pair.sector().nup() - _secA[isect].nup();
        int ndownB = pair.sector().ndown() - _secA[isect].ndown();
        if(
         (nupB < 0) ||
         (ndownB < 0) ||
         (nupB > _Ns_B) ||
         (ndownB > _Ns_B)
        ){
         continue;
        }
        sector _secB = sector(nupB, ndownB,
          _symB[0].comb().c_n_k(_Ns_B, nupB) * _symB[0].comb().c_n_k(_Ns_B, ndownB)
        );
        _symB[0].set_sector(_secB);
        for(size_t ii = 0; ii < _secB.size(); ++ii){
          _symB[0].next_state();
          long long stateB = _symB[0].state();
          _symA[0].set_sector(_secA[isect]);
          for(size_t jj = 0; jj < _secA[isect].size(); ++jj){
            _symA[0].next_state();
            long long state0 = mergestate(_symA[0].state(), stateB);
            _symA[1].set_sector(_secA[isect]);
            for(size_t kk = 0; kk < _secA[isect].size(); ++kk){
              _symA[1].next_state();
              long long state1 = mergestate(_symA[1].state(), stateB);
              _rho[isect][jj][kk] += weight *
#ifdef USE_MPI
                evec[_ham.model().symmetry().index(state0)] *
                evec[_ham.model().symmetry().index(state1)];
#else
                pair.eigenvector()[_ham.model().symmetry().index(state0)] *
                pair.eigenvector()[_ham.model().symmetry().index(state1)];
#endif
            }
          }
        }
      }
    }

    /**
     * Combine basis vectors.
     *
     * @param  stateA basis vector of subsystem A
     * @param  stateB basis vector of subsystem B
     * @return        basis vector of the whole system
     */
    long long mergestate(long long stateA, long long stateB){
      long long state = 0;
      long long newstate = 0;
      int isign;
      for(size_t ispin = 0; ispin < _Nspins; ++ispin){
        for(size_t iorb = 0; iorb < _orbsA.size(); ++iorb){
          if(_ham.model().checkState(stateA, iorb + ispin * _Ns_A, _Nspins * _Ns_A)){
            _ham.model().adag(_orbsA[iorb]  + ispin * _Ns, state, newstate, isign);
            state = newstate;
          }
        }
        for(size_t iorb = 0; iorb < _orbsB.size(); ++iorb){
          if(_ham.model().checkState(stateB, iorb + ispin * _Ns_B, _Nspins * _Ns_B)){
            _ham.model().adag(_orbsB[iorb]  + ispin * _Ns, state, newstate, isign);
            state = newstate;
          }
        }
      }
      return state;
    }

    void invalidate_cache(){
      _eigenvalues_valid = false;
      _entanglement_entropy_valid = false;
      _quadratic_entropy_valid = false;
    }

    /// The hamiltonian of the system A+B
    Hamiltonian& _ham;
    /// Reduced density matrix for the system A
    std::map<size_t, std::vector<std::vector<precision>>> _rho;

    /// Cached observables and their validity.
    std::vector<precision> _eigenvalues;
    bool _eigenvalues_valid;
    precision _entanglement_entropy;
    bool _entanglement_entropy_valid;
    precision _quadratic_entropy;
    bool _quadratic_entropy_valid;

    /// Symmetries of the system A.
    std::vector<symmetry> _symA;
    /// Symmetry of the system B.
    // FIXME Have to use a vector even for one symmetry, else the constructor won't compile.
    std::vector<symmetry> _symB;
    /// Symmetry sectors of the system A.
    std::vector<sector> _secA;
    /// The orbitals for which the density matrix is calculated.
    std::vector<size_t> _orbsA;
    std::vector<size_t> _orbsB;
    /// The number of sites.
    size_t _Ns, _Ns_A, _Ns_B;
    size_t _Ip;
    size_t _Nspins;
    /// Inverse temperature.
    precision _beta;
    /// Minimal Boltzmann-factor.
    precision _cutoff;

  };

}

#endif
