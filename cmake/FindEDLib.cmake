
include(FindPackageHandleStandardArgs)


find_path(EDLib_INCLUDE_DIR NAMES "edlib/Hamiltonian.h" HINTS ${EDLib_DIR}/include)

find_package_handle_standard_args(EDLib DEFAULT_MSG EDLib_INCLUDE_DIR)

mark_as_advanced(EDLib_INCLUDE_DIR)