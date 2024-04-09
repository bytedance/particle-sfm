find_library(OpenMP_LIBRARY
    NAMES omp5 gomp
)


find_path(OpenMP_INCLUDE_DIR
    omp.h
    #HINTS /usr/lib/gcc/x86_64-linux-gnu/4.9/include/
    HINTS /usr/lib/gcc/x86_64-linux-gnu/8/include/
    HINTS /usr/lib/gcc/x86_64-linux-gnu/9/include/
    HINTS /usr/lib/gcc/x86_64-linux-gnu/10/include/
    HINTS /usr/lib/gcc/x86_64-linux-gnu/11/include/
    HINTS /usr/lib/gcc/x86_64-linux-gnu/12/include/
)

mark_as_advanced(OpenMP_LIBRARY OpenMP_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
if(IS_GNU)
  find_package_handle_standard_args(OpenMP DEFAULT_MSG 
      OpenMP_INCLUDE_DIR)
  set(OpenMP_LIBRARY "")
else()
  find_package_handle_standard_args(OpenMP DEFAULT_MSG 
      OpenMP_LIBRARY OpenMP_INCLUDE_DIR)
endif()

if (OpenMP_FOUND)
    set(OpenMP_INCLUDE_DIRS ${OpenMP_INCLUDE_DIR})
    set(OpenMP_COMPILE_OPTIONS -Xpreprocessor -fopenmp)
    set(OpenMP_libomp_LIBRARY ${OpenMP_LIBRARY})
    set(OpenMP_LIBRARIES ${OpenMP_LIBRARY})

    if(IS_GNU)
    else()
      add_library(OpenMP::OpenMP SHARED IMPORTED)
      set_target_properties(OpenMP::OpenMP PROPERTIES
          IMPORTED_LOCATION ${OpenMP_LIBRARIES}
          INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIRS}"
          INTERFACE_COMPILE_OPTIONS "${OpenMP_COMPILE_OPTIONS}"
      )
    endif()
endif()

