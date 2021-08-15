#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "benchmark::benchmark" for configuration "Release"
set_property(TARGET benchmark::benchmark APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(benchmark::benchmark PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/benchmark.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS benchmark::benchmark )
list(APPEND _IMPORT_CHECK_FILES_FOR_benchmark::benchmark "${_IMPORT_PREFIX}/lib/benchmark.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
