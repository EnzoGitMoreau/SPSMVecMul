CXX = g++
CC = gcc

OPENBLAS_DIR = /opt/homebrew/Cellar/openblas/0.3.27
OPENMP_DIR = /opt/Homebrew/Cellar/libomp/18.1.6

CURRENT_DIR = $(shell pwd)
EXEC = tests
CXXFLAGS = -std=c++11 -fopenmp 

INCLUDES = -Iinclude/ \
	-Iboost_1_84_0/\
	-I/usr/include/aarch64-linux-gnu/\
	-I$(OPENBLAS_DIR)\
	-I$(GraphBLAS) \
	-I$(OPENMP_DIR)/lib \
	-I$(ARM_PER_LIB)/include
	
       
SRC_PATH = src
LIB_DIRS = -L -Lboost_1_84_0/stage/lib -L/usr/local/lib -L$(OPENMP_DIR)/lib 
LIBS = -lboost_thread -lopenblas
SRCS = $(SRC_PATH)/main.cpp \
 $(SRC_PATH)/sparmatsymblk.cc \
 $(SRC_PATH)/MatSymBMtInstance.cpp \
 

OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)
TARGET = $(EXEC)

all:
	$(MAKE) clean
	$(MAKE) compile
	$(MAKE) install
	
compile: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_DIRS) -o $(TARGET) $(OBJS) $(LIBS)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o : %.cc
	$(CXX) $(CCFLAGS) $(INCLUDES) -c $< -o $@

profile:
	$(MAKE) clean
	$(MAKE) all CXXFLAGS="$(CXXFLAGS) -pg -O3"

	
install:
	
	   
clean:
	rm -f $(OBJS) $(TARGET)

