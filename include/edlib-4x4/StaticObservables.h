#ifndef E4X4_STATICOBSERVABLES_H
#define E4X4_STATICOBSERVABLES_H

#include "edlib/StaticObservables.h"

namespace E4x4 {
  template<class Hamiltonian>
  class StaticObservables : public EDLib::StaticObservables<Hamiltonian>{

  public:

    /**
     * @brief Print largest coefficients of an eigenvector in decreasing order of magnitude.
     *
     * @param ham - the Hamiltonian
     * @param pair - the eigenpair
     * @param nmax - maximum number of coefficients to be processed;
     * @param trivial - skip the coefficients smaller than this number.
     */
    void print_largest_coefficients(Hamiltonian& ham, const EDLib::EigenPair<precision, sector>& pair, size_t nmax, precision trivial){
      std::vector<std::pair<long long, precision>> coeffs = find_largest_coefficients(ham, pair, nmax, trivial);
#ifdef USE_MPI
      int myid;
      MPI_Comm_rank(ham.comm(), &myid);
      if(!myid)
#endif
      {
        std::cout << "Eigenvector components for eigenvalue " << pair.eigenvalue() << " ";
        pair.sector().print();
        std::cout << std::endl;
        for(size_t i = 0; i < coeffs.size(); ++i){
          std::cout << coeffs[i].second << " * |";
          std::string spin_down = std::bitset< 64 >( coeffs[i].first ).to_string().substr(64-  ham.model().orbitals(), ham.model().orbitals());
          std::string spin_up   = std::bitset< 64 >( coeffs[i].first ).to_string().substr(64-2*ham.model().orbitals(), ham.model().orbitals());
          std::cout<<spin_up<< "|"<<spin_down;
          std::cout << ">" << std::endl;
        }
      }
    }

  };

}

#endif
