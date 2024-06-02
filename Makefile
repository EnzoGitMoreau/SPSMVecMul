CXX = /usr/bin/clang++
CC = /usr/bin/clang

OPENBLAS_DIR = /opt/homebrew/Cellar/openblas/0.3.27
OPENMP_DIR = /opt/homebrew/Cellar/libomp/18.1.6
ARM_PER_LIB = /opt/arm/armpl_24.04_flang-new_clang_18
BOOST_DIR = /opt/homebrew/Cellar/boost/1.85.0

CURRENT_DIR = $(shell pwd)
EXEC = tests
CXXFLAGS := -std=c++17 -Xclang -fopenmp -O3

INCLUDES = -Iinclude/ \
	-I$(BOOST_DIR)/include\
	-I$(OPENBLAS_DIR)\
	-I$(OPENMP_DIR)/include \
	-I$(ARM_PER_LIB)/include
	
       
SRC_PATH = src
LIB_DIRS = -L$(OPENBLAS_DIR)/lib -L$(BOOST_DIR)/lib -L/usr/local/lib -L$(OPENMP_DIR)/lib -L$(ARM_PER_LIB)/lib
LIBS = -lboost_thread-mt -lopenblasp-r0.3.27  -larmpl_lp64 -lomp
SRCS = $(SRC_PATH)/main.cpp \
 $(SRC_PATH)/sparmatsymblk.cc \
 $(SRC_PATH)/MatSymBMtInstance.cpp \
 $(SRC_PATH)/matsym.cc

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
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

profile:
	$(MAKE) clean
	$(MAKE) all CXXFLAGS="$(CXXFLAGS) -pg -O3"

	
install:
	install_name_tool -add_rpath $(CURRENT_DIR)/bin/boost_1_84_0/stage/lib $(EXEC) && \
    install_name_tool -add_rpath /usr/local/lib $(EXEC)&& \
    install_name_tool -add_rpath $(OPENMP_DIR)/lib $(EXEC)
	install_name_tool -add_rpath $(ARM_PER_LIB)/lib $(EXEC)
clean:
	rm -f $(OBJS) $(TARGET)

