project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++1z)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(source_no_main src/particle_filter.cpp)
set(sources ${source_no_main} src/main.cpp)

set(test_name test_main)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1.11.0/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 


add_executable(particle_filter ${sources})


target_link_libraries(particle_filter z ssl uv uWS)

# GoogleTest
# Download and unpack googletest at configure time
configure_file(thirdparty/googletest/CMakeLists.txt.in thirdparty/googletest/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/thirdparty/googletest)
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/thirdparty/googletest)
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

# Now simply link against gtest or gtest_main as needed. Eg
include_directories(src)
file(GLOB test_src_files ${PROJECT_SOURCE_DIR}/tests/*.cpp)
add_executable(${test_name} ${test_src_files})
set_target_properties(${test_name} PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
target_link_libraries(${test_name} gtest_main)
enable_testing()
add_test(testrun ${test_name} --gtest_color=yes)
