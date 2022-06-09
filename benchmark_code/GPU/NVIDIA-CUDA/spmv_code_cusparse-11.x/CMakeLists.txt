cmake_minimum_required (VERSION 3.10)
#set(CMAKE_VERBOSE_MAKEFILE on)

#	c) auto

#	d) auto

## 2) CUDA PREFIX
set (CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")

# 	a) Compiler
set (CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
set (CMAKE_CXX_COMPILER ${CXX_PREFIX}/bin/c++)
set (CMAKE_C_COMPILER ${CXX_PREFIX}/bin/gcc)

# 	b) Compile flags
set (CMAKE_CUDA_FLAGS "-w -O3 -arch=sm_80") #35 for Tesla K40, 70 for V100, 61 for GTX 1060/70, 80 for A100
message("CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})

#	c) Architecture 
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

## 4) ITER run for benchmark results
add_definitions(-DNR_ITER=128)

## 5) Define full DDEBUG
#add_definitions(-DDDEBUG)

## 6) Define result testing flag
#add_definitions(-DTEST)

project (MakeSpMVGreatAgain11 CXX CUDA)
set ( PROJECT_HEAD_DIR ${PROJECT_SOURCE_DIR}/../../../ )
set ( PROJECT_TEST_DIR ${PROJECT_SOURCE_DIR}/tests )
set ( PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src )
message( "PROJECT_HEAD_DIR: " ${PROJECT_HEAD_DIR} )
message( "PROJECT_TEST_DIR: " ${PROJECT_TEST_DIR} )
message( "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
message( "PROJECT_BINARY_DIR: " ${PROJECT_BINARY_DIR} )
message( "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
message( "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

##Mpakos Python flags
#execute_process(COMMAND python3-config --cflags OUTPUT_VARIABLE MPAKOS_PYTHON_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
#message( "MPAKOS_PYTHON_CFLAGS: " ${MPAKOS_PYTHON_CFLAGS} )

##Mpakos C flags
#set (MPAKOS_CFLAGS "-g -Wall -Wno-unused-result -Wsign-compare -Wformat -Werror=format-security -DNDEBUG -fwrapv -fstack-protector" )
#message( "MPAKOS_CFLAGS: " ${MPAKOS_CFLAGS} )

# C++ compile flags
#set(CMAKE_CXX_STANDARD 11) # For dungani
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fopenmp -lm")# ${MPAKOS_PYTHON_CFLAGS}")
message( "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

#execute_process(COMMAND python3-config --embed --ldflags OUTPUT_VARIABLE PYTHON_LINK OUTPUT_STRIP_TRAILING_WHITESPACE)
#set(PYTHON_LINK "${PYTHON_LINK} -Xlinker -export-dynamic")
#message( "PYTHON_LINK: " ${PYTHON_LINK} )

## CUDA libs and includes
set (CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
include_directories(${CUDA_INCLUDE_DIRS} )
message( "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS} )
set (CUDA_LD "${CUDA_TOOLKIT_ROOT_DIR}/lib64,${CUDA_TOOLKIT_ROOT_DIR}/lib")
set (CUDA_LINK "-lcudart -lcublas -lcusparse") # -L${CUDA_LD} 
message( "CUDA_LINK: " ${CUDA_LINK} )

set(NVEMAPI_PREFIX "/home/panastas/PhD_stuff/SpMV-Research/benchmark_code/GPU/nvidia-energy-measure")
set (NVEMAPI_INCLUDE_DIRS ${NVEMAPI_PREFIX}/include) # -L${CUDA_LD} 
include_directories(${NVEMAPI_INCLUDE_DIRS} )
message( "NVEMAPI_INCLUDE_DIRS: " ${NVEMAPI_INCLUDE_DIRS} )

set (NVEMAPI_LINK "-L${CUDA_LD} -lnvidia-ml -L/usr/lib64 -lcuda -lpthread") # ${PYTHON_LINK}
message( "NVEMAPI_LINK: " ${NVEMAPI_LINK} )

# My header files 
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/inc)
message( "My header files: " ${CMAKE_CURRENT_SOURCE_DIR}/inc )

# Mpakos header files 
#set (MPAKOS_PREFIX "${PROJECT_HEAD_DIR}/artificial_matrix_generation/pmpakos_impl")
#include_directories( ${MPAKOS_PREFIX}/include)
#message( "Mpakos header files: " ${MPAKOS_PREFIX}/include )

# DGAL header files 
set (DGAL_PREFIX "${PROJECT_HEAD_DIR}/artificial-matrix-generator")
include_directories( ${DGAL_PREFIX})
message( "DGAL header files: " ${DGAL_PREFIX} )

set (INTERNAL_LINK_LIB "${CUDA_LINK} ${NVEMAPI_LINK} -lrt -ldl") #${PYTHON_LINK} 
message( "INTERNAL_LINK_LIB: " ${INTERNAL_LINK_LIB})

set (LINK_LIB "${CUDA_LINK} -lrt -ldl ${NVEMAPI_LINK}") #${PYTHON_LINK} 
message( "LINK_LIB: " ${LINK_LIB} )

file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.c* ${NVEMAPI_PREFIX}/src/*.*)
message( "SOURCE_FILES: " ${SOURCE_FILES} )

file(GLOB DGAL_FILES ${DGAL_PREFIX}/*.c*)
message( "DGAL_FILES: " ${DGAL_FILES} )

## datatype used in benchmarks. 0 = float, 1 = double
set(DATATYPE 1) 

if (${DATATYPE} EQUAL 1)
	#add_library(MyLib ${SOURCE_FILES} ${MPAKOS_PREFIX}/include/artificial_matrix_generation_double.o)
	#add_library(MyLib ${SOURCE_FILES} ${DGAL_PREFIX}/artificial_matrix_generation_double.so)
	set (DGAL_LINK "-L${DGAL_PREFIX} -lartificial_matrix_generation_double")
	add_library(MyLib ${SOURCE_FILES})
	target_link_libraries(MyLib ${INTERNAL_LINK_LIB} ${DGAL_LINK})
	add_definitions(-DVALUE_TYPE_AX=double)
	add_definitions(-DVALUE_TYPE_Y=double)
	add_definitions(-DVALUE_TYPE_COMP=double)

	add_executable (cuSPARSEDcsrmv_11-0_mtx ${PROJECT_TEST_DIR}/cuSPARSEcsrmv_11-0_mtx.cu)
	target_link_libraries(cuSPARSEDcsrmv_11-0_mtx ${LINK_LIB} MyLib)
	add_executable (cuSPARSEDcoomv_11-0_mtx ${PROJECT_TEST_DIR}/cuSPARSEcoomv_11-0_mtx.cu)
	target_link_libraries(cuSPARSEDcoomv_11-0_mtx ${LINK_LIB} MyLib)

	add_executable (cuSPARSEDcsrmv_11-0_generate ${PROJECT_TEST_DIR}/cuSPARSEcsrmv_11-0_generate.cu)
	target_link_libraries(cuSPARSEDcsrmv_11-0_generate ${LINK_LIB} MyLib)
	add_executable (cuSPARSEDcoomv_11-0_generate ${PROJECT_TEST_DIR}/cuSPARSEcoomv_11-0_generate.cu)
	target_link_libraries(cuSPARSEDcoomv_11-0_generate ${LINK_LIB} MyLib)
elseif(${DATATYPE} EQUAL 0)
	#add_library(MyLib ${SOURCE_FILES} ${MPAKOS_PREFIX}/include/artificial_matrix_generation_float.o)
	#add_library(MyLib ${SOURCE_FILES} ${DGAL_PREFIX}/artificial_matrix_generation_float.so)
	set (DGAL_LINK "-L${DGAL_PREFIX} -lartificial_matrix_generation_float")
	add_library(MyLib ${SOURCE_FILES})
	target_link_libraries(MyLib ${INTERNAL_LINK_LIB} ${DGAL_LINK})
	add_definitions(-DVALUE_TYPE_AX=float)
	add_definitions(-DVALUE_TYPE_Y=float)
	add_definitions(-DVALUE_TYPE_COMP=float)
	add_executable (cuSPARSEScsrmv_11-0_mtx ${PROJECT_TEST_DIR}/cuSPARSEcsrmv_11-0_mtx.cu)
	target_link_libraries(cuSPARSEScsrmv_11-0_mtx ${LINK_LIB} MyLib)
	add_executable (cuSPARSEScoomv_11-0_mtx ${PROJECT_TEST_DIR}/cuSPARSEcoomv_11-0_mtx.cu)
	target_link_libraries(cuSPARSEScoomv_11-0_mtx ${LINK_LIB} MyLib)


	add_executable (cuSPARSEScsrmv_11-0_generate ${PROJECT_TEST_DIR}/cuSPARSEcsrmv_11-0_generate.cu)
	target_link_libraries(cuSPARSEScsrmv_11-0_generate ${LINK_LIB} MyLib)
	add_executable (cuSPARSEScoomv_11-0_generate ${PROJECT_TEST_DIR}/cuSPARSEcoomv_11-0_generate.cu)
	target_link_libraries(cuSPARSEScoomv_11-0_generate ${LINK_LIB} MyLib)
else()

#add_definitions(-DVALUE_TYPE_AX=int8_t)
#add_definitions(-DVALUE_TYPE_Y=int32_t)
#add_definitions(-DVALUE_TYPE_COMP=int32_t)
#add_executable (cuSPARSEMIcsrmv_11-0_mtx ${PROJECT_TEST_DIR}/cuSPARSEcsrmv_11-0_mtx.cu)
#target_link_libraries(cuSPARSEMIcsrmv_11-0_mtx ${LINK_LIB} MyLib)
endif()
