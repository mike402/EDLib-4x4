#include <iostream>

#include <edlib/EDParams.h>
#include "edlib/Hamiltonian.h"
#include "edlib/SzSymmetry.h"
#include "edlib/SOCRSStorage.h"
#include "edlib/CRSStorage.h"
#include "edlib/HubbardModel.h"
#include "edlib/GreensFunction.h"
#include "edlib/ChiLoc.h"
#include "edlib/HDF5Utils.h"
#include "edlib/SpinResolvedStorage.h"
#include "edlib/StaticObservables.h"
#include "edlib-4x4/DensityMatrix.h"
#include "edlib-4x4/PairingSusceptibility.h"
#include "edlib/MeshFactory.h"

int main(int argc, const char ** argv) {
#ifdef USE_MPI
  typedef EDLib::SRSHubbardHamiltonian HamType;
#else
  typedef EDLib::SOCSRHubbardHamiltonian HamType;
#endif
#ifdef USE_MPI
  MPI_Init(&argc, (char ***) &argv);
#endif
  alps::params params(argc, argv);
  if(params.help_requested(std::cout)) {
    exit(0);
  }
  EDLib::define_parameters(params);
  alps::hdf5::archive ar;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(!rank)
#endif
  ar.open(params["OUTPUT_FILE"].as<std::string>().c_str(), "w");
  try {
#ifdef USE_MPI
    HamType ham(params, MPI_COMM_WORLD);
#else
    HamType ham(params);
#endif
    ham.diag();
    EDLib::StaticObservables<HamType> so(params);
    so.print_static_observables(ham);
    for (const auto& pair :ham.eigenpairs()) {
      so.print_major_electronic_configuration(ham, pair, 256, 1e-5);
      so.print_class_contrib(ham, pair, 256, 1e-5, true);
    }
    std::vector<EDLib::DensityMatrix<HamType>> dm;
    for(size_t ii = 0; ii < params["NSITES"]; ++ii){
      dm.push_back(EDLib::DensityMatrix<HamType>(params, ham, std::set<size_t> {ii}));
    }
    std::vector<std::vector<EDLib::DensityMatrix<HamType>>> dmAB;
    for(size_t ii = 0; ii < params["NSITES"]; ++ii){
      std::vector<EDLib::DensityMatrix<HamType>> tmp;
      for(size_t jj = 0; jj < params["NSITES"]; ++jj){
        tmp.push_back(EDLib::DensityMatrix<HamType>(params, ham, std::set<size_t> {ii, jj}));
      }
      dmAB.push_back(tmp);
    }
    for(size_t ii = 0; ii < dm.size(); ++ii){
      dm[ii].compute();
    }
#ifdef USE_MPI
    if(!rank)
#endif
    {
      std::cout << "S_quad:" << std:: endl;
      for(size_t ii = 0; ii < dm.size(); ++ii){
        std::cout << dm[ii].quadratic_entropy() << std::endl;
      }
      std::cout << "S_ent:" << std::endl;
      for(size_t ii = 0; ii < dm.size(); ++ii){
        std::cout << dm[ii].entanglement_entropy() << std::endl;
      }
    }
    for(size_t ii = 0; ii < dmAB.size(); ++ii){
      for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
        dmAB[ii][jj].compute();
      }
    }
#ifdef USE_MPI
    if(!rank)
#endif
    {
      std::cout << "S_quad_AB:" << std::endl;
      for(size_t ii = 0; ii < dmAB.size(); ++ii){
        for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
          if(jj){
            std::cout << "\t";
          }
          std::cout << dmAB[ii][jj].quadratic_entropy();
        }
        std::cout << std::endl;
      }
      std::cout << "I_quad_AB:" << std::endl;
      for(size_t ii = 0; ii < dmAB.size(); ++ii){
        for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
          if(jj){
            std::cout << "\t";
          }
          std::cout << dm[ii].quadratic_entropy() + dm[jj].quadratic_entropy() - dmAB[ii][jj].quadratic_entropy();
        }
        std::cout << std::endl;
      }
      std::cout << "S_ent_AB:" << std::endl;
      for(size_t ii = 0; ii < dmAB.size(); ++ii){
        for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
          if(jj){
            std::cout << "\t";
          }
          std::cout << dmAB[ii][jj].entanglement_entropy();
        }
        std::cout << std::endl;
      }
      std::cout << "I_ent_AB:" << std::endl;
      for(size_t ii = 0; ii < dmAB.size(); ++ii){
        for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
          if(jj){
            std::cout << "\t";
          }
          std::cout << dm[ii].entanglement_entropy() + dm[jj].entanglement_entropy() - dmAB[ii][jj].entanglement_entropy();
        }
        std::cout << std::endl;
      }
    }
/*
*/
    EDLib::hdf5::save_eigen_pairs(ham, ar, "results");
/*
    EDLib::gf::PairingSusceptibility < HamType, alps::gf::real_frequency_mesh> psusc(params, ham, std::vector<std::array<size_t, 2>> {{0, 1}});
    psusc.compute();
    psusc.save(ar, "results");
*/
    //EDLib::gf::GreensFunction < HamType, alps::gf::matsubara_positive_mesh, alps::gf::statistics::statistics_type> greensFunction(params, ham,alps::gf::statistics::statistics_type::FERMIONIC);
    EDLib::gf::GreensFunction < HamType, alps::gf::real_frequency_mesh> greensFunction(params, ham);
    greensFunction.compute();
    greensFunction.save(ar, "results");
    //EDLib::gf::ChiLoc<HamType, alps::gf::matsubara_positive_mesh, alps::gf::statistics::statistics_type> susc(params, ham, alps::gf::statistics::statistics_type::BOSONIC);
    EDLib::gf::ChiLoc< HamType, alps::gf::real_frequency_mesh> susc(params, ham);
    susc.compute();
    susc.save(ar, "results");
    susc.compute<EDLib::gf::NOperator<double> >();
    susc.save(ar, "results");
  } catch (std::exception & e) {
#ifdef USE_MPI
    if(!rank) std::cerr<<e.what()<<std::endl;
#else
    std::cerr<<e.what();
#endif
  }
#ifdef USE_MPI
  if(!rank)
#endif
  ar.close();
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
