##### Overview
EDLib-4x4 is an exact diagonalization solver built with the EDLib library.

At present EDLib-4x4 uses the features of the forked EDLib.
https://github.com/mike402/EDLib

##### Installation ###
To compile the program and tests create a build directory and run

1. `cmake -DARPACK_DIR=<path to ARPACK-ng library dir> -DEDLib_DIR=<path to EDLib installation> -DTesting=ON {path_to_edlib}`
2. `make`
3. `make test` (for running tests)

To build with MPI support add `-DUSE_MPI=ON` *CMake* flag. *MPI* library should be installed and *ALPSCore*
library should be compiled with *MPI* support. To build with a specific *ALPSCore* library use
`-DALPSCore_DIR=<path to ALPSCore>` *CMake* flag. Since the critical for current library implementation
MPI-related *ARPACK-ng* bug was recenlty fixed it is stricly recommended to use the latest version
of *ARPACK-ng* from github repository.

##### Dependencies
- c++11-compatible compiler (tested with clang >= 3.1, gcc >= 4.8.2, icpc >= 14.0.2)
- *ALPSCore* library >= 0.5.6-alpha3
- *arpack-ng* >= 3.5.0
- *MPI* standard >= 2.1 (optional)
- *git* to fetch the code
- *cmake* to build tests and examples (optional)

##### Authors
- Sergei Iskakov, *iskakoff[at]q-solvers.ru*, 2017-now
- Michael Danilov, 2017-now

##### Distribution
Open-source under MIT License.
