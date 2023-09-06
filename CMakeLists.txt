cmake_minimum_required(VERSION 3.10)

project(min_grad CXX)

set(CAFFE2_USE_CUDNN 1)
#set(BUILD_SHARED_LIBS 0)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

#set(CMAKE_CXX_FLAGS_RELEASE "/MT")
#set(CMAKE_CXX_FLAGS_DEBUG "/MTd")


set(CMAKE_PREFIX_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
	)
find_package(Torch REQUIRED)


set(ASSIMP_INSTALL OFF)
set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_HUNTER_ENABLED OFF)
set(BUILD_SHARED_LIBS OFF)
set(ASSIMP_NO_EXPORT ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/assimp)

file(GLOB SOURCES
	${SOURCE_DIR}/*.cpp
	)

file(GLOB HEADERS
	${INCLUDE_DIR}/*.h
	${THIRD_PARTY_DIR}/stb/*.h
	)

add_executable(min_grad ${SOURCES} ${HEADERS})


target_link_libraries(min_grad PRIVATE
	${TORCH_LIBRARIES}
	assimp::assimp
	)

target_include_directories(min_grad PRIVATE
	${INCLUDE_DIR}
	${THIRD_PARTY_DIR}/stb
	)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)

target_compile_features(min_grad PUBLIC cxx_std_20)

enable_language(CXX)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi -DDEBUG")

set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_FLAGS "/openmp")
add_definitions(-D_UNICODE)

 if (MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   list(APPEND ${TORCH_DLLS} ${TORCH_PDBS})
   add_custom_command(TARGET min_grad
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:min_grad>)
 endif (MSVC)