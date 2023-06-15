cmake_minimum_required(VERSION 3.0)
project(GCOLMAP)

################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

################################################################################
# Options
################################################################################

option(SIMD_ENABLED "Whether to enable SIMD optimizations" ON)
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)
option(IPO_ENABLED "Whether to enable interprocedural optimization" ON)
option(PROFILING_ENABLED "Whether to enable google-perftools linker flags" OFF)

################################################################################
# Find packages
################################################################################

if(OPENMP_ENABLED)
    find_package(OpenMP REQUIRED)
endif()

find_package(Theia REQUIRED)
find_package(COLMAP REQUIRED)
message(STATUS "Ceres version: ${CERES_VERSION}")
if(${CERES_VERSION} VERSION_LESS "2.2.0")
    # ceres 2.2.0 changes the interface of local parameterization
    add_definitions("-DCERES_PARAMETERIZATION_ENABLED")
endif()

################################################################################
# Compiler specific configuration
################################################################################

if(CMAKE_BUILD_TYPE)
    message(STATUS "Build type specified as ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type not specified, using Release")
    set(CMAKE_BUILD_TYPE Release)
    set(IS_DEBUG OFF)
endif()

if(IS_MSVC)
    # Some fixes for the Glog library.
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if(IS_GNU)
    # Hide incorrect warnings for uninitialized Eigen variables under GCC.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-sign-compare")
endif()

if(IS_DEBUG)
    add_definitions("-DEIGEN_INITIALIZE_MATRICES_BY_NAN")
endif()

if(SIMD_ENABLED)
    message(STATUS "Enabling SIMD support")
else()
    message(STATUS "Disabling SIMD support")
endif()

if(OPENMP_ENABLED AND OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Disabling OpenMP support")
endif()

if(IPO_ENABLED AND NOT IS_DEBUG AND NOT IS_GNU)
    message(STATUS "Enabling interprocedural optimization")
    set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION 1)
else()
    message(STATUS "Disabling interprocedural optimization")
endif()

if(PROFILING_ENABLED)
    message(STATUS "Enabling profiling support")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lprofiler -ltcmalloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler -ltcmalloc")
else()
    message(STATUS "Disabling profiling support")
endif()

################################################################################
# Add sources
################################################################################

# Generate source file with version definitions.
set(GCOLMAP_INCLUDE_DIRS
    ${COLMAP_INCLUDE_DIRS}
    ${THEIA_INCLUDE_DIRS}
)

set(GCOLMAP_LINK_DIRS
    ${COLMAP_LINK_DIRS}
)

set(GCOLMAP_EXTERNAL_LIBRARIES
    ${CMAKE_DL_LIBS}
    ${COLMAP_LIBRARIES}
    ${THEIA_LIBRARIES}
)

if(OPENMP_FOUND)
	message(STATUS "OpenMP: ${OpenMP_libomp_LIBRARY}")
	list(APPEND GCOLMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif()

if(UNIX)
    list(APPEND GCOLMAP_EXTERNAL_LIBRARIES pthread)
endif()

include_directories(
    src
    ${GCOLMAP_INCLUDE_DIRS}
)
link_directories(${GCOLMAP_LINK_DIRS})

add_subdirectory(src)

################################################################################
# Generate source groups for Visual Studio, XCode, etc.
################################################################################

GCOLMAP_ADD_SOURCE_DIR(src/base BASE_SRCS *.h *.cc)
GCOLMAP_ADD_SOURCE_DIR(src/controllers CONTROLLERS_SRCS *.h *.cc)
GCOLMAP_ADD_SOURCE_DIR(src/estimators ESTIMATORS_SRCS *.h *.cc)
GCOLMAP_ADD_SOURCE_DIR(src/exe EXE_SRCS *.h *.cc)
GCOLMAP_ADD_SOURCE_DIR(src/optim OPTIM_SRCS *.h *.cc)
GCOLMAP_ADD_SOURCE_DIR(src/sfm SFM_SRCS *.h *.cc)
GCOLMAP_ADD_SOURCE_DIR(src/util UTIL_SRCS *.h *.cc)
GCOLMAP_ADD_SOURCE_DIR(src/global GLOBAL_SRCS *.h *.cc)

# Add all of the source files to a regular library target, as using a custom
# target does not allow us to set its C++ include directories (and thus
# intellisense can't find any of the included files).
add_library(
    ${GCOLMAP_SRC_ROOT_FOLDER}
    ${BASE_SRCS}
    ${CONTROLLERS_SRCS}
    ${ESTIMATORS_SRCS}
    ${EXE_SRCS}
    ${OPTIM_SRCS}
    ${SFM_SRCS}
    ${UTIL_SRCS}
    ${GLOBAL_SRCS}
)

# Prevent the library from being compiled automatically.
set_target_properties(
    ${GCOLMAP_SRC_ROOT_FOLDER} PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1)

################################################################################
# Install and uninstall scripts
################################################################################

# Install header files.
install(DIRECTORY src/
        DESTINATION include/gcolmap
        FILES_MATCHING PATTERN "*.h")

# Generate and install CMake configuration.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfig.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/GCOLMAPConfig.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GCOLMAPConfig.cmake"
        DESTINATION "share/gcolmap")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeConfigVersion.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/GCOLMAPConfigVersion.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GCOLMAPConfigVersion.cmake"
        DESTINATION "share/gcolmap")

# Install find_package scripts for dependencies.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
        DESTINATION share/gcolmap
        FILES_MATCHING PATTERN "Find*.cmake")

# Install batch scripts under Windows.
if(IS_MSVC)
    install(FILES "scripts/shell/GCOLMAP.bat"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION "/")
endif()

# Install application meny entry under Linux/Unix.
if(UNIX AND NOT APPLE)
    install(FILES "doc/GCOLMAP.desktop" DESTINATION "share/applications")
endif()

# Configure the uninstallation script.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeUninstall.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake"
               IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)
set_target_properties(uninstall PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})

