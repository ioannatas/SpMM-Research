.PHONY: all clean

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables

# Targets that don't generate dependency files.
NODEPS = clean

DIRS = obj

define Rule_Auto_Dependencies_base =
    $(1:.o=.d): $(2) | $(DIRS)
	@echo 'Generating dependencies file:  $(2)'
	gcc $$(CFLAGS) -MT '$(1:.o=.d)' -MM -MG '$(2)' -MF '$(1:.o=.d)'
    ifeq (0, $(words $(findstring $(MAKECMDGOALS),$(NODEPS))))
        -include $(1:.o=.d)
    endif
    $(1): $(1:.o=.d)
endef

define Rule_Auto_Dependencies =
    $(eval $(call Rule_Auto_Dependencies_base,$(1),$(2)))
    $(1): $(2)
endef

CPATH = 

library = ../../lib


CC = gcc
# CC = clang

CPP = g++
# CPP = clang++

ARCH = $(shell uname -m)

CFLAGS = -Wall -Wextra
# Tells the compiler to use pipes instead of temporary files (faster compilation, but uses more memory).
CFLAGS += -pipe
CFLAGS += -fopenmp
ifeq ($(ARCH), x86_64)
    CFLAGS += -mbmi
    CFLAGS += -mbmi2
    CFLAGS += -march=native
	CFLAGS += -mavx2
endif
# CFLAGS += -g3 -fno-omit-frame-pointer
# CFLAGS += -Og
# CFLAGS += -O0
# CFLAGS += -O2
CFLAGS += -O3

# CFLAGS += -ffast-math

# CFLAGS += -flto
# CFLAGS += -march=native

CFLAGS += -D'LEVEL1_DCACHE_LINESIZE=$(shell getconf LEVEL1_DCACHE_LINESIZE)'
CFLAGS += -D'LEVEL1_DCACHE_SIZE=$(shell getconf LEVEL1_DCACHE_SIZE)'
CFLAGS += -D'LEVEL2_CACHE_SIZE=$(shell getconf LEVEL2_CACHE_SIZE)'
CFLAGS += -D'LEVEL3_CACHE_SIZE=$(shell getconf LEVEL3_CACHE_SIZE)'

CFLAGS += -I'$(library)'


# DOUBLE := 0
DOUBLE := 1

CFLAGS += -D'DOUBLE=$(DOUBLE)'

ifeq ($(DOUBLE), 1)
	CFLAGS += -D'ValueType=double'
else
	CFLAGS += -D'ValueType=float'
endif


LDFLAGS =
LDFLAGS += -lm


LIB_SRC = read_mtx.cpp pthread_functions.c omp_functions.c string_util.c io.c random.c parallel_io.c hash.c array_metrics.c plot.c csr.c csc.c kmeans.c kmeans_char.c

LIB_OBJ := $(LIB_SRC)
LIB_OBJ := $(patsubst %.c,obj/%.o,$(LIB_OBJ))
LIB_OBJ := $(patsubst %.cpp,obj/%.o,$(LIB_OBJ))


# EXE = mat_feat.exe mat_experiment.exe mat_experiment_char.exe mat_experiment_inverse.exe mat_experiment_char_inverse.exe mat_reorder.exe
# EXE = mat_feat.exe mat_experiment.exe mat_experiment_char.exe mat_experiment_col_sorting.exe 
EXE = mat_feat.exe mat_experiment_col_sorting.exe 
# mat_experiment_high_low_splitting.exe mat_experiment_col_row_sorting.exe mat_experiment_row_col_sorting.exe

all: $(EXE) | $(DIRS)


%.exe: obj/%.o $(LIB_OBJ)
	$(CPP) $(CFLAGS) $^ -o $@ $(LDFLAGS)


$(call Rule_Auto_Dependencies,obj/mat_feat.o,mat_feat.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_experiment.o,mat_experiment.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_experiment_char.o,mat_experiment_char.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_experiment_inverse.o,mat_experiment_inverse.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_experiment_char_inverse.o,mat_experiment_char_inverse.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_reorder.o,mat_reorder.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_experiment_high_low_splitting.o,mat_experiment_high_low_splitting.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_experiment_col_row_sorting.o,mat_experiment_col_row_sorting.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_experiment_col_sorting.o,mat_experiment_col_sorting.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/mat_experiment_row_col_sorting.o,mat_experiment_row_col_sorting.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@

$(call Rule_Auto_Dependencies,obj/read_mtx.o,read_mtx.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@

$(call Rule_Auto_Dependencies,obj/pthread_functions.o,$(library)/pthread_functions.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/omp_functions.o,$(library)/omp_functions.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/string_util.o,$(library)/string_util.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/io.o,$(library)/io.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/parallel_io.o,$(library)/parallel_io.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/hash.o,$(library)/hash/hash.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/random.o,$(library)/random.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/array_metrics.o,$(library)/array_metrics.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/plot.o,$(library)/plot/plot.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/csr.o,csr.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/csc.o,csc.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/kmeans.o,$(library)/kmeans/kmeans.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/kmeans_char.o,$(library)/kmeans/kmeans_char.c)
	$(CC) $(CFLAGS) -c $< -o $@

$(DIRS): %:
	mkdir -p $@

clean:
	$(RM) obj/*.o obj/*.d *.o *.exe a.out

