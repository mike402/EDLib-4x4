#ifndef E4X4_PAIRINGSUSCEPTIBILITY_H
#define E4X4_PAIRINGSUSCEPTIBILITY_H


#include <iomanip>
#include "edlib/Lanczos.h"
#include "edlib/EigenPair.h"
#include "edlib/ExecutionStatistic.h"

namespace EDLib {
  namespace gf {
    template<class Hamiltonian, typename Mesh, typename... Args>
    class PairingSusceptibility : public Lanczos < Hamiltonian, Mesh, Args...> {
      using Lanczos < Hamiltonian, Mesh, Args... >::hamiltonian;
      using Lanczos < Hamiltonian, Mesh, Args... >::lanczos;
      using Lanczos < Hamiltonian, Mesh, Args... >::omega;
      using Lanczos < Hamiltonian, Mesh, Args... >::beta;
      using Lanczos < Hamiltonian, Mesh, Args... >::compute_sym_continued_fraction;
      using typename Lanczos < Hamiltonian, Mesh, Args... >::precision;
    public:
      PairingSusceptibility(alps::params &p, Hamiltonian &h, std::vector<std::array<size_t, 2>> gf_orbs, Args ... args) : Lanczos < Hamiltonian, Mesh, Args... >(p, h, args...), _model(h.model()),
                                                        gf(Lanczos < Hamiltonian, Mesh, Args... >::omega(), alps::gf::index_mesh(gf_orbs.size())),
                                                        _gf_orbs(gf_orbs),
                                                        _cutoff(p["lanc.BOLTZMANN_CUTOFF"]) {
        if(p["storage.EIGENVALUES_ONLY"] == 1) {
          throw std::logic_error("Eigenvectors have not been computed. Green's function can not be evaluated.");
        }
      }

      void compute() {
        gf *= 0.0;
        _Z = 0.0;
        if(hamiltonian().eigenpairs().empty())
          return;
#ifdef USE_MPI
        int rank;
        MPI_Comm_rank(hamiltonian().storage().comm(), &rank);
#endif
        /// get groundstate
        const EigenPair<precision, typename Hamiltonian::ModelType::Sector> &groundstate =  *hamiltonian().eigenpairs().begin();
        /// compute statsum
        for (auto kkk = hamiltonian().eigenpairs().begin(); kkk != hamiltonian().eigenpairs().end(); kkk++) {
          const EigenPair<precision, typename Hamiltonian::ModelType::Sector> &eigenpair = *kkk;
          _Z += std::exp(-(eigenpair.eigenvalue() - groundstate.eigenvalue()) * beta());
        }
        common::statistics.registerEvent("Greens function");
        /// iterate over eigen-pairs
        for (auto kkk = hamiltonian().eigenpairs().begin(); kkk != hamiltonian().eigenpairs().end(); kkk++) {
          const EigenPair<precision, typename Hamiltonian::ModelType::Sector>& pair = *kkk;
          /// compute Boltzmann-factor
          precision boltzmann_f = std::exp(-(pair.eigenvalue() - groundstate.eigenvalue()) * beta());
          /// Skip all eigenvalues with Boltzmann-factor smaller than cutoff
          if (boltzmann_f < _cutoff) {
//        std::cout<<"Skipped by Boltzmann factor."<<std::endl;
            continue;
          }
#ifdef USE_MPI
          if(rank==0)
#endif
          std::cout << "Compute Green's function contribution for eigenvalue E=" << pair.eigenvalue() << " with Boltzmann factor = " << boltzmann_f << "; for sector" << pair.sector() << std::endl;
          local_contribution(pair, groundstate);
        }
#ifdef USE_MPI
        if(rank == 0) {
#endif
        /// normalize Green's function by statsum Z.
        gf /= _Z;
#ifdef USE_MPI
        }
#endif
        common::statistics.updateEvent("Greens function");
      }

      /**
       * Save Green's function in the hdf5 archive and in plain text file
       * @param ar -- hdf5 archive to save Green's function
       * @param path -- root path in hdf5 archive
       */
      void save(alps::hdf5::archive& ar, const std::string & path) {
#ifdef USE_MPI
        int rank;
        MPI_Comm_rank(hamiltonian().storage().comm(), &rank);
        if(rank == 0) {
#endif
          gf.save(ar, path + "/PSusc_omega");
          std::ostringstream Gomega_name;
          Gomega_name << "PSusc_omega";
          std::ofstream G_omega_file(Gomega_name.str().c_str());
          G_omega_file << std::setprecision(14) << gf;
          G_omega_file.close();
          std::cout << "Statsum: " << _Z << std::endl;
          ar[path + "/@Statsum"] << _Z;
#ifdef USE_MPI
        }
#endif
      }

      void compute_selfenergy(alps::hdf5::archive &ar, const std::string &path){
        alps::gf::three_index_gf<std::complex<double>, Mesh, alps::gf::index_mesh, alps::gf::index_mesh> bare(gf.mesh1(), gf.mesh2(), gf.mesh3());
        alps::gf::three_index_gf<std::complex<double>, Mesh, alps::gf::index_mesh, alps::gf::index_mesh> sigma(gf.mesh1(), gf.mesh2(), gf.mesh3());
        _model.bare_greens_function(bare, beta());
        bare.save(ar, path + "/G0_omega");
        std::ostringstream Gomega_name;
        Gomega_name << "G0_omega";
        std::ofstream G_omega_file(Gomega_name.str().c_str());
        G_omega_file << std::setprecision(14) << bare;
        G_omega_file.close();
        for(int iw = 0; iw< bare.mesh1().points().size(); ++iw) {
          typename Mesh::index_type w(iw);
          for (int im: bare.mesh2().points()) {
            for (int is : bare.mesh3().points()) {
              sigma(w, alps::gf::index_mesh::index_type(im), alps::gf::index_mesh::index_type(is)) =
                1.0/bare(w, alps::gf::index_mesh::index_type(im), alps::gf::index_mesh::index_type(is)) - 1.0/gf(w, alps::gf::index_mesh::index_type(im), alps::gf::index_mesh::index_type(is));
            }
          }
        }
        sigma.save(ar, path + "/Sigma_omega");
        Gomega_name.str("");
        Gomega_name << "Sigma_omega";
        G_omega_file.open(Gomega_name.str().c_str());
        G_omega_file << std::setprecision(14) << sigma;
        G_omega_file.close();
      }

    private:

      /**
       * Compute local Green's function G_ii
       *
       * @param groundstate -- system groundstate
       * @param pair -- current Eigen-Pair
       */
      void local_contribution(const EigenPair<precision, typename Hamiltonian::ModelType::Sector>& pair, const EigenPair<precision, typename Hamiltonian::ModelType::Sector>& groundstate) {
#ifdef USE_MPI
        int rank;
        MPI_Comm_rank(hamiltonian().storage().comm(), &rank);
#endif
        /// iterate over orbitals for the diagonal Green's function
        for (int iorb = 0; iorb < _gf_orbs.size(); ++iorb) {
          std::array<size_t, 2> orbs = _gf_orbs[iorb];
          std::vector < precision > outvec(1, precision(0.0));
          precision expectation_value = 0.0;
          _model.symmetry().set_sector(pair.sector());
          /// first we are going to create particle and compute contribution to Green's function
          if (create_two_particles(orbs, pair.eigenvector(), outvec, expectation_value)) {
            /// Perform Lanczos factorization for starting vector |outvec>
            int nlanc = lanczos(outvec);
#ifdef USE_MPI
            if(!rank)
#endif
            {
              std::cout << "orbitals: " << orbs[0] << ", " << orbs[1] << " <n|aa*|n>=" << expectation_value << " nlanc:" << nlanc << std::endl;
              /// Using computed Lanczos factorization compute approximation for \frac{1}{z - H} by calculation of a continued fraction
              compute_sym_continued_fraction(expectation_value, pair.eigenvalue(), groundstate.eigenvalue(), nlanc, 1, gf, index_mesh_index(iorb));
            }
          }
          /// restore symmetry sector
          _model.symmetry().set_sector(pair.sector());
          /// perform the same for destroying of a particle
          if (annihilate_two_particles(orbs, pair.eigenvector(), outvec, expectation_value)) {
            int nlanc = lanczos(outvec);
#ifdef USE_MPI
            if(!rank)
#endif
            {
              std::cout << "orbitals: " << orbs[0] << ", " << orbs[1] << " <n|a*a|n>=" << expectation_value << " nlanc:" << nlanc << std::endl;
              compute_sym_continued_fraction(expectation_value, pair.eigenvalue(), groundstate.eigenvalue(), nlanc, -1, gf, index_mesh_index(iorb));
            }
          }
        }
      }

      /// Green's function type
      typedef alps::gf::two_index_gf<std::complex<double>, Mesh, alps::gf::index_mesh >  GF_TYPE;
      typedef typename alps::gf::index_mesh::index_type index_mesh_index;
      typedef typename Mesh::index_type frequency_mesh_index;
      /// Green's function container object
      GF_TYPE gf;
      /// Model we are solving
      typename Hamiltonian::ModelType &_model;
      /// Boltzmann-factor cutoff
      precision _cutoff;
      /// Statsum
      precision _Z;
      /// Orbital pairs to calculate the Green's function.
      std::vector<std::array<size_t, 2>> _gf_orbs;

      /**
       * @brief Perform the create operator action to the eigenstate
       *
       * @param orbitals - the orbitals to create the particles, first with the spin down.
       * @param invec - current eigenstate
       * @param outvec - Op-vec product
       * @param expectation_value - expectation value of aa*
       * @return true if the particle has been created
       */
      bool create_two_particles(std::array<size_t, 2> orbitals, const std::vector < precision > &invec, std::vector < precision > &outvec, double &expectation_value) {
        // check that the particle can be annihilated
        if ((_model.symmetry().sector().nup() == _model.orbitals()) || (_model.symmetry().sector().ndown() == _model.orbitals())) {
          return false;
        }
        hamiltonian().storage().reset();
        long long k = 0;
        int sign = 0;
        int nup_tmp = _model.symmetry().sector().nup();
        int ndn_tmp = _model.symmetry().sector().ndown() + 1;
        typename Hamiltonian::ModelType::Sector tmp_sec(nup_tmp, ndn_tmp, _model.symmetry().comb().c_n_k(_model.orbitals(), nup_tmp) * _model.symmetry().comb().c_n_k(_model.orbitals(), ndn_tmp));
        std::vector<precision> tmpvec(hamiltonian().storage().vector_size(tmp_sec), 0.0);
        common::statistics.registerEvent("adag");
        hamiltonian().storage().a_adag(orbitals[0] + _model.orbitals(), invec, tmpvec, tmp_sec, false);
        common::statistics.updateEvent("adag");
        std::cout<<"adag in "<<common::statistics.event("adag").first<<" s \n";
        int nup_new = nup_tmp + 1;
        int ndn_new = ndn_tmp;
        typename Hamiltonian::ModelType::Sector next_sec(nup_new, ndn_new, _model.symmetry().comb().c_n_k(_model.orbitals(), nup_new) * _model.symmetry().comb().c_n_k(_model.orbitals(), ndn_new));
        outvec.assign(hamiltonian().storage().vector_size(next_sec), 0.0);
        common::statistics.registerEvent("adag");
        hamiltonian().storage().a_adag(orbitals[1], tmpvec, outvec, next_sec, false);
        common::statistics.updateEvent("adag");
        std::cout<<"adag in "<<common::statistics.event("adag").first<<" s \n";
        double norm = hamiltonian().storage().vv(outvec, outvec);
        for (int j = 0; j < outvec.size(); ++j) {
          outvec[j] /= std::sqrt(norm);
        }
        std::cout<<"norm" << norm << std::endl;
        _model.symmetry().set_sector(next_sec);
        expectation_value = norm;
        return true;
      };

      /**
       * @brief Perform the annihilator operator action to the eigenstate
       *
       * @param orbitals - the orbitals to destroy the particles, first with the spin down.
       * @param invec - current eigenstate
       * @param outvec - Op-vec product
       * @param expectation_value - expectation value of a*a
       * @return true if the particle has been destroyed
       */
      bool annihilate_two_particles(std::array<size_t, 2> orbitals, const std::vector < precision > &invec, std::vector < precision > &outvec, double &expectation_value) {
        // check that the particle can be annihilated
        if ((_model.symmetry().sector().nup() == 0) || (_model.symmetry().sector().ndown() == 0)) {
          return false;
        }
        hamiltonian().storage().reset();
        long long k = 0;
        int sign = 0;
        int nup_tmp = _model.symmetry().sector().nup();
        int ndn_tmp = _model.symmetry().sector().ndown() - 1;
        typename Hamiltonian::ModelType::Sector tmp_sec(nup_tmp, ndn_tmp, _model.symmetry().comb().c_n_k(_model.orbitals(), nup_tmp) * _model.symmetry().comb().c_n_k(_model.orbitals(), ndn_tmp));
        std::vector<precision> tmpvec(hamiltonian().storage().vector_size(tmp_sec), precision(0.0));
        common::statistics.registerEvent("a");
        hamiltonian().storage().a_adag(orbitals[0] + _model.orbitals(), invec, tmpvec, tmp_sec, true);
        common::statistics.updateEvent("a");
        std::cout<<"a in "<<common::statistics.event("a").first<<" s \n";
        int nup_new = nup_tmp - 1;
        int ndn_new = ndn_tmp;
        typename Hamiltonian::ModelType::Sector next_sec(nup_new, ndn_new, _model.symmetry().comb().c_n_k(_model.orbitals(), nup_new) * _model.symmetry().comb().c_n_k(_model.orbitals(), ndn_new));
        outvec.assign(hamiltonian().storage().vector_size(next_sec), precision(0.0));
        common::statistics.registerEvent("a");
        hamiltonian().storage().a_adag(orbitals[1], tmpvec, outvec, next_sec, true);
        common::statistics.updateEvent("a");
        std::cout<<"a in "<<common::statistics.event("a").first<<" s \n";
        double norm = hamiltonian().storage().vv(outvec, outvec);
        for (int j = 0; j < outvec.size(); ++j) {
          outvec[j] /= std::sqrt(norm);
        }
        std::cout<<"norm" << norm << std::endl;
        _model.symmetry().set_sector(next_sec);
        // <v|a^{\star}a|v>
        expectation_value = norm;
        return true;
      };
    };
  }
}

#endif //E4X4_PAIRINGSUSCEPTIBILITY_H
