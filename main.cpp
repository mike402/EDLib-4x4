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
    EDLib::DensityMatrix<HamType> dm(params, ham, std::vector<size_t> {0});
/*
    EDLib::StaticObservables<HamType> so(params);
    so.print_static_observables(ham);
    for (const auto& pair :ham.eigenpairs()) {
      so.print_largest_coefficients(ham, pair, 256, 1e-5);
      so.print_class_contrib(ham, pair, 256, 1e-5, true);
    }
*/
    dm.compute();
    dm.print();
#ifdef USE_MPI
    if(!rank){
#endif
     std::cout << "Entanglement spectrum:" << std::endl;
     std::vector<double> espec = dm.eigenvalues();
     for(size_t ii = 0; ii < espec.size(); ++ii){
       std::cout << espec[ii] << std::endl;
     }
     std::cout << "Tr(rho - rho^2) = " << dm.quadratic_entropy() << std::endl;
     std::cout << "S_ent = " << dm.entanglement_entropy() << std::endl;
    }
    EDLib::hdf5::save_eigen_pairs(ham, ar, "results");
/*
    EDLib::gf::PairingSusceptibility < HamType, alps::gf::real_frequency_mesh> psusc(params, ham, std::vector<std::array<size_t, 2>> {{0, 1}});
    psusc.compute();
    psusc.save(ar, "results");
    //EDLib::gf::GreensFunction < HamType, alps::gf::matsubara_positive_mesh, alps::gf::statistics::statistics_type> greensFunction(params, ham,alps::gf::statistics::statistics_type::FERMIONIC);
    EDLib::gf::GreensFunction < HamType, alps::gf::real_frequency_mesh> greensFunction(params, ham, std::set<std::array<size_t, 2>> {{5, 5}});
    greensFunction.compute();
    greensFunction.save(ar, "results");
    //EDLib::gf::ChiLoc<HamType, alps::gf::matsubara_positive_mesh, alps::gf::statistics::statistics_type> susc(params, ham, alps::gf::statistics::statistics_type::BOSONIC);
    EDLib::gf::ChiLoc< HamType, alps::gf::real_frequency_mesh> susc(params, ham, std::set<std::array<size_t, 2>> {{5, 5}});
    susc.compute();
    susc.save(ar, "results");
    susc.compute<EDLib::gf::NOperator<double> >();
    susc.save(ar, "results");
*/
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
