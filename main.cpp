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
    std::vector<EDLib::DensityMatrix<HamType>> dm;
    size_t Nsites = size_t(params["NSITES"]);
    for(size_t ii = 0; ii < Nsites; ++ii){
      dm.push_back(EDLib::DensityMatrix<HamType>(params, ham, std::set<size_t> {ii}));
    }
    std::vector<std::vector<EDLib::DensityMatrix<HamType>>> dmAB;
    for(size_t ii = 0; ii < Nsites; ++ii){
      std::vector<EDLib::DensityMatrix<HamType>> tmp;
      for(size_t jj = 0; jj < Nsites; ++jj){
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
      std::ofstream S_file("S.txt");
      for(size_t ii = 0; ii < dm.size(); ++ii){
        S_file << dm[ii].entanglement_entropy() << std::endl;
      }
      S_file.close();
/*
      std::ofstream S_quad_file("S_quad.txt");
      for(size_t ii = 0; ii < dm.size(); ++ii){
        S_quad_file << dm[ii].quadratic_entropy() << std::endl;
      }
      S_quad_file.close();
*/
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
      std::ofstream S_AB_file("S_AB.txt");
      for(size_t ii = 0; ii < dmAB.size(); ++ii){
        for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
          if(jj){
            S_AB_file << "\t";
          }
          S_AB_file << dmAB[ii][jj].entanglement_entropy();
        }
        S_AB_file << std::endl;
      }
      S_AB_file.close();
      std::ofstream I_AB_file("I_AB.txt");
      for(size_t ii = 0; ii < dmAB.size(); ++ii){
        for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
          if(jj){
            I_AB_file << "\t";
          }
          I_AB_file << 0.5 * (dm[ii].entanglement_entropy() + dm[jj].entanglement_entropy() - dmAB[ii][jj].entanglement_entropy());
        }
        I_AB_file << std::endl;
      }
      I_AB_file.close();
/*
      std::ofstream S_quad_AB_file("S_quad_AB.txt");
      for(size_t ii = 0; ii < dmAB.size(); ++ii){
        for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
          if(jj){
            S_quad_AB_file << "\t";
          }
          S_quad_AB_file << dmAB[ii][jj].quadratic_entropy();
        }
        S_quad_AB_file << std::endl;
      }
      S_quad_AB_file.close();
      std::ofstream I_quad_AB_file("I_quad_AB.txt");
      for(size_t ii = 0; ii < dmAB.size(); ++ii){
        for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
          if(jj){
            I_quad_AB_file << "\t";
          }
          I_quad_AB_file << 0.5 * (dm[ii].quadratic_entropy() + dm[jj].quadratic_entropy() - dmAB[ii][jj].quadratic_entropy());
        }
        I_quad_AB_file << std::endl;
      }
      I_quad_AB_file.close();
*/
    }
/*
    EDLib::DensityMatrix<HamType> dm1(params, ham);
#ifdef USE_MPI
    if(!rank){
#endif
     std::ofstream dm1_file("entanglement_spectrum.txt");
     std::vector<double> dmevals = dm1.eigenvalues();
     for(size_t ii = 0; ii < dmevals.size(); ++ii){
      dm1_file << dm1.eigenvalues()[ii] << std::endl;
     }
     dm1_file.close();
     std::cout << "S: " << dm1.entanglement_entropy() << std::endl;
     std::cout << "S_quad: " << dm1.quadratic_entropy() << std::endl;
#ifdef USE_MPI
    }
#endif
    EDLib::hdf5::save_eigen_pairs(ham, ar, "results");
*/
/*
    EDLib::gf::PairingSusceptibility < HamType, alps::gf::real_frequency_mesh> psusc(params, ham, std::vector<std::array<size_t, 2>> {{0, 1}});
    psusc.compute();
    psusc.save(ar, "results");
    //EDLib::gf::GreensFunction < HamType, alps::gf::matsubara_positive_mesh, alps::gf::statistics::statistics_type> greensFunction(params, ham,alps::gf::statistics::statistics_type::FERMIONIC);
    EDLib::gf::GreensFunction < HamType, alps::gf::real_frequency_mesh> greensFunction(params, ham);
    greensFunction.compute();
    greensFunction.save(ar, "results");
#ifdef USE_MPI
    if(!rank){
#endif
      std::ostringstream Gomega_g_name, Gomega_l_name;
      Gomega_g_name << "G_g_omega_r";
      std::ofstream G_omega_g_file(Gomega_g_name.str().c_str());
      G_omega_g_file << std::setprecision(14) << greensFunction.G_g();
      G_omega_g_file.close();
      Gomega_l_name << "G_l_omega_r";
      std::ofstream G_omega_l_file(Gomega_l_name.str().c_str());
      G_omega_l_file << std::setprecision(14) << greensFunction.G_l();
      G_omega_l_file.close();
      std::ostringstream Gomega_g_name2, Gomega_l_name2;
      Gomega_g_name2 << "G_g_ij_omega_r";
      std::ofstream G_omega_g_file2(Gomega_g_name2.str().c_str());
      G_omega_g_file2<< std::setprecision(14) << greensFunction.G_g_ij();
      G_omega_g_file2.close();
      Gomega_l_name2 << "G_l_ij_omega_r";
      std::ofstream G_omega_l_file2(Gomega_l_name2.str().c_str());
      G_omega_l_file2<< std::setprecision(14) << greensFunction.G_l_ij();
      G_omega_l_file2.close();
#ifdef USE_MPI
    }
#endif
*/
    EDLib::StaticObservables<HamType> so(params);
    std::map<std::string, std::vector<double>> observables = so.calculate_static_observables(ham);
    so.print_static_observables(ham, std::cout);
/*
    for (const auto& pair :ham.eigenpairs()) {
      so.print_major_electronic_configuration(ham, pair, 256, 1e-5, so_out);
      so.print_class_contrib(ham, pair, 256, 1e-5, true, so_out);
    }
*/
/*
    //EDLib::gf::ChiLoc<HamType, alps::gf::matsubara_positive_mesh, alps::gf::statistics::statistics_type> susc(params, ham, alps::gf::statistics::statistics_type::BOSONIC);
    EDLib::gf::ChiLoc< HamType, alps::gf::real_frequency_mesh> susc(params, ham);
    // compute average magnetic moment
    double avg = 0.0;
    for(auto x : observables[so._M_]) {
      avg += x / (2.0*observables[so._M_].size());
    }
    // compute spin susceptibility
    susc.compute<EDLib::gf::SzOperator<double>>(&avg);
    susc.save(ar, "results");
    // compute average occupancy moment
    avg = 0.0;
    for(auto x : observables[so._N_]) {
      avg += x / double(observables[so._N_].size());
    }
    // Compute sharge susceptibility
    susc.compute<EDLib::gf::NOperator<double>>(&avg);
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
