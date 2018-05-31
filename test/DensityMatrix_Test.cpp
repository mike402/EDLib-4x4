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
  alps::params p;
  EDLib::define_parameters(p);
  p["NSITES"]=2;
  p["NSPINS"]=2;
  p["INPUT_FILE"]="test/input/dimer/input.h5";
  p["arpack.SECTOR"]=false;
  p["storage.MAX_SIZE"]=256;
  p["storage.MAX_DIM"]=16;
  p["arpack.NEV"]=1024;
  p["lanc.BETA"]=10000.0;

#ifdef USE_MPI
  typedef EDLib::SRSHubbardHamiltonian HamType;
#else
  typedef EDLib::SOCSRHubbardHamiltonian HamType;
#endif
  HamType ham(p
#ifdef USE_MPI
  , MPI_COMM_WORLD
#endif
  );

  ham.diag();

  // Reduced density matrix for one site.
  EDLib::DensityMatrix<HamType> dm(p, ham, "DensityMatrix_orbitals");
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
  EDLib::DensityMatrix<HamType> fulldm(p, ham, "FullDensityMatrix_orbitals");
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
  alps::params p;
  EDLib::define_parameters(p);
  p["NSITES"]=4;
  p["NSPINS"]=2;
  p["INPUT_FILE"]="test/input/plaquette/input.h5";
  p["arpack.SECTOR"]=false;
  p["storage.MAX_SIZE"]=65536;
  p["storage.MAX_DIM"]=256;
  p["arpack.NEV"]=1024;
  p["lanc.BETA"]=10000.0;

#ifdef USE_MPI
  typedef EDLib::SRSHubbardHamiltonian HamType;
#else
  typedef EDLib::SOCSRHubbardHamiltonian HamType;
#endif
  HamType ham(p
#ifdef USE_MPI
  , MPI_COMM_WORLD
#endif
  );

  ham.diag();

/*
EDLib::StaticObservables<HamType> so(p);
so.print_static_observables(ham);
for (const auto& pair :ham.eigenpairs()) {
  so.print_major_electronic_configuration(ham, pair, 65535, 1e-5);
}
*/

  // Check that full density matrix for one eigenvector equals
  EDLib::DensityMatrix<HamType> fulldm(p, ham, "FullDensityMatrix_orbitals");
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

// FIXME Compute reduced density matrices naively and compare with them

  // Reduced density matrices for two sites.
  std::vector<EDLib::DensityMatrix<HamType>> dm;
  std::vector<std::vector<double>> dmspec;
  for(size_t ii = 0; ii < 4; ++ii){
    std::ostringstream name;
    name << "DensityMatrix" << ii << "_orbitals";
    dm.push_back(EDLib::DensityMatrix<HamType>(p, ham, name.str().c_str()));
    dm[ii].compute();
//dm[ii].printfull();
    ASSERT_EQ(dm[ii].full().size(), 16);
    ASSERT_EQ(dm[ii].full()[0].size(), 16);
//for(size_t jj = 0; jj < dm[ii].eigenvalues().size(); ++jj){
//  std::cout << dm[ii].eigenvalues()[jj] << std::endl;
//}
  }
  // Check that density matrix for equivalent site pairs is the same.
  ASSERT_NEAR(dm[0].quadratic_entropy(), dm[1].quadratic_entropy(), 1e-14);
  ASSERT_NEAR(dm[0].entanglement_entropy(), dm[1].entanglement_entropy(), 1e-14);
  ASSERT_NEAR(dm[0].entanglement_entropy(), dm[1].entanglement_entropy(), 1e-14);
  ASSERT_NEAR(dm[2].quadratic_entropy(), dm[3].quadratic_entropy(), 1e-14);
  ASSERT_NEAR(dm[2].entanglement_entropy(), dm[3].entanglement_entropy(), 1e-14);
  ASSERT_NEAR(dm[2].entanglement_entropy(), dm[3].entanglement_entropy(), 1e-14);
/*
  for(size_t ii = 0; ii < dm[0].size(); ++ii){
   for(size_t jj = 0; jj < dm[0][0].size(); ++jj){
     std::vector<double> spectrum = dm[0].eigenvalues();
     std::vector<double> spectrum = dm[0].eigenvalues();
     std::vector<double> spectrum = dm[2].eigenvalues();
     ASSERT_NEAR(dm[0][ii][jj], dm[1][ii][jj], 1e-5);
     ASSERT_NEAR(dm[2][ii][jj], dm[3][ii][jj], 1e-5);
   }
  }
*/
}
