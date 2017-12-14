#include <gtest/gtest.h>
#include "edlib/Hamiltonian.h"
#include "edlib/HubbardModel.h"
#include "edlib-4x4/DensityMatrix.h"
#include "edlib/Storage.h"
#include "edlib/EDParams.h"


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
  EDLib::DensityMatrix<HamType> dm(p, ham, std::vector<size_t> {0});
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
  EDLib::DensityMatrix<HamType> fulldm(p, ham, std::vector<size_t> {0, 1});
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
