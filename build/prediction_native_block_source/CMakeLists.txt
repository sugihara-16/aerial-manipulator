cmake_minimum_required(VERSION 3.16)
project(drive_interval_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(pinocchio 3.8 REQUIRED)
add_library(drive_interval SHARED interval.cpp)
target_include_directories(drive_interval PRIVATE ../../third_party/nlohmann_json)
target_link_libraries(drive_interval PRIVATE pinocchio::pinocchio)
target_compile_options(drive_interval PRIVATE -O3 -Wall -Wextra)
# Instantiate the numerical kernels here so they use the same CPU options as
# the interval loop. SciPy already supplies a single-threaded BLAS dependency;
# its prefixed LP64 symbols avoid mixing with NumPy's separate ILP64 library.
execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python -c
  "import pathlib,scipy; print(next((pathlib.Path(scipy.__file__).parent.parent/'scipy.libs').glob('libscipy_openblas-*.so')))"
  OUTPUT_VARIABLE scipy_blas OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
target_link_libraries(drive_interval PRIVATE ${scipy_blas})
target_compile_options(drive_interval PRIVATE -UPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION -DEIGEN_USE_BLAS
  -Ddgemm_=scipy_dgemm_ -Ddgemv_=scipy_dgemv_ -Ddsyrk_=scipy_dsyrk_ -Ddtrsm_=scipy_dtrsm_)
