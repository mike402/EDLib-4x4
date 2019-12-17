#include <gtest/gtest.h>
#include "edlib/Hamiltonian.h"
#include "edlib/HubbardModel.h"
#include "edlib-4x4/DensityMatrix.h"
#include "edlib/Storage.h"
#include "edlib/EDParams.h"

#include "edlib/StaticObservables.h"


#ifdef USE_MPI

class HubbardModelTestEnv : public ::testing::Environment {
  protected:
  virtual void SetUp() {
    char** argv;
    int argc = 0;
    int mpiError = MPI_Init(&argc, &argv);
  }

  virtual void TearDown() {
    MPI_Finalize();
  }

  ~HubbardModelTestEnv(){};

};

::testing::Environment* const foo_env = AddGlobalTestEnvironment(new HubbardModelTestEnv);

#endif


TEST(HubbardModelTest, ReferenceTest) {
  alps::params params;
  EDLib::define_parameters(params);
  params["NSITES"]=2;
  params["NSPINS"]=2;
  params["INPUT_FILE"]="test/input/dimer/input.h5";
  params["arpack.SECTOR"]=true;
  params["storage.MAX_SIZE"]=256;
  params["storage.MAX_DIM"]=16;
  params["arpack.NEV"]=1024;
  params["lanc.BETA"]=10000.0;

#ifdef USE_MPI
  typedef EDLib::SRSHubbardHamiltonian HamType;
#else
  typedef EDLib::SOCSRHubbardHamiltonian HamType;
#endif
  HamType ham(params
#ifdef USE_MPI
  , MPI_COMM_WORLD
#endif
  );

  ham.diag();

  // Reduced density matrix for one site.
  EDLib::DensityMatrix<HamType> dm(params, ham, "DensityMatrix_orbitals");
  dm.compute();
  std::vector<std::vector<double>> rho = dm.full();
  ASSERT_EQ(rho.size(), 4);
  ASSERT_EQ(rho[0].size(), 4);
  for(size_t ii = 0; ii < 2; ++ii){
    for(size_t jj = 0; jj < 2; ++jj){
      if(ii != jj){
        // There must be zeroes outside of the sectors.
        ASSERT_EQ(rho[ii][jj], 0.0);
      }else if (ii && (ii < 2)){
        ASSERT_NEAR(rho[ii][jj], 0.5, 1e-5);
      }else{
        ASSERT_NEAR(rho[ii][jj], 0.0, 1e-5);
      }
    }
  }

  // Full density matrix.
  EDLib::DensityMatrix<HamType> fulldm(params, ham, "FullDensityMatrix_orbitals");
  fulldm.compute();
  std::vector<std::vector<double>> fullrho = fulldm.full();
  ASSERT_EQ(fullrho.size(), 16);
  ASSERT_EQ(fullrho[0].size(), 16);
  std::vector<double> espec = fulldm.eigenvalues();
  ASSERT_NEAR(espec[0], 1.0, 1e-9);
  for(size_t ii = 1; ii < espec.size(); ++ii){
    ASSERT_NEAR(espec[ii], 0.0, 1e-6);
  }
  ASSERT_NEAR(fulldm.quadratic_entropy(), 0.0, 1e-9);
  ASSERT_NEAR(fulldm.entanglement_entropy(), 0.0, 1e-9);

}

TEST(HubbardModelTest2, ReferenceTest) {
  alps::params params;
  EDLib::define_parameters(params);
  params["NSITES"]=4;
  params["NSPINS"]=2;
  params["INPUT_FILE"]="test/input/plaquette/input.h5";
  params["arpack.SECTOR"]=false;
  params["storage.MAX_SIZE"]=65536;
  params["storage.MAX_DIM"]=256;
  params["arpack.NEV"]=1024;
  params["lanc.BETA"]=10000.0;

#ifdef USE_MPI
  typedef EDLib::SRSHubbardHamiltonian HamType;
#else
  typedef EDLib::SOCSRHubbardHamiltonian HamType;
#endif
  HamType ham(params
#ifdef USE_MPI
  , MPI_COMM_WORLD
#endif
  );

  ham.diag();

#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  // Check that full density matrix for one eigenvector equals that computed the basic way.
  EDLib::DensityMatrix<HamType> fulldm(params, ham, "FullDensityMatrix_orbitals");
  fulldm.compute();
  auto pair = ham.eigenpairs().begin();
  for(size_t isect = 0; isect < fulldm.sectors().size(); ++isect) {
    if((fulldm.sectors()[isect].nup() == pair->sector().nup()) && (fulldm.sectors()[isect].ndown() == pair->sector().ndown())){
      for(size_t ii = 0; ii < pair->eigenvector().size(); ++ii){
        for(size_t jj = 0; jj < pair->eigenvector().size(); ++jj){
          ASSERT_NEAR(fulldm.matrix().at(isect)[ii][jj], pair->eigenvector()[ii] * pair->eigenvector()[jj], 1e-9);
        }
      }
    }
  }

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
    std::vector<double> spectrum = dm[ii].eigenvalues();
  }
  for(size_t ii = 0; ii < dmAB.size(); ++ii){
    for(size_t jj = 0; jj < dmAB[0].size(); ++jj){
      dmAB[ii][jj].compute();
      std::vector<double> spectrum = dmAB[ii][jj].eigenvalues();
    }
  }

  // Check that density matrices of all sites are the same.
  std::vector<double> spectrum0 = dm[0].eigenvalues();
  for(int iorb = 1; iorb < dm.size(); ++iorb){
    ASSERT_EQ(dm[0].full().size(), dm[iorb].full().size());
    std::vector<double> spectrum1 = dm[iorb].eigenvalues();
    for(size_t ii = 0; ii < spectrum0.size(); ++ii){
     ASSERT_NEAR(spectrum0[ii], spectrum1[ii], 1e-15);
    }
    ASSERT_NEAR(dm[0].quadratic_entropy(), dm[iorb].quadratic_entropy(), 1e-15);
    ASSERT_NEAR(dm[0].entanglement_entropy(), dm[iorb].entanglement_entropy(), 1e-15);
  }

  // Check that density matrices of equivalent site pairs are the same.
  std::vector<std::vector<std::vector<int>>> equiv = {
   {{ 0, 1}, { 2, 3}, { 0, 2}, { 1, 3}, { 1, 0}, { 3, 2}, { 2, 0}, { 3, 1}},
   {{ 0, 3}, { 1, 2}, { 3, 0}, { 2, 1}, {-1,-1}, {-1,-1}, {-1,-1}, {-1,-1}},
  };
  for(int igrp = 0; igrp < equiv.size(); ++igrp){
    int ind0_0 = equiv[igrp][0][0];
    int ind0_1 = equiv[igrp][0][1];
    for(int ieq = 1; ieq < equiv[igrp].size(); ++ieq){
      int ind1_0 = equiv[igrp][ieq][0];
      int ind1_1 = equiv[igrp][ieq][1];
      if((ind1_0 == -1) || (ind1_1 == -1)){
        continue;
      }
      ASSERT_EQ(dmAB[ind0_0][ind0_1].full().size(), dmAB[ind1_0][ind1_1].full().size());
      std::vector<double> spectrum0 = dmAB[ind0_0][ind0_1].eigenvalues();
      std::vector<double> spectrum1 = dmAB[ind1_0][ind1_1].eigenvalues();
      for(size_t ii = 0; ii < spectrum0.size(); ++ii){
       ASSERT_NEAR(spectrum0[ii], spectrum1[ii], 1e-14);
      }
      ASSERT_NEAR(dmAB[ind0_0][ind0_1].quadratic_entropy(), dmAB[ind1_0][ind1_1].quadratic_entropy(), 1e-14);
      ASSERT_NEAR(dmAB[ind0_0][ind0_1].entanglement_entropy(), dmAB[ind1_0][ind1_1].entanglement_entropy(), 1e-14);
    }
  }

}
