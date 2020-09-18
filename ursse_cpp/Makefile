PYTHON_VERSION := 3.6
PYTHON_INC := /usr/include/python$(PYTHON_VERSION)
BOOST_INC := /usr/include/boost
XTENSOR_INC := /home/ilobach/anaconda3/include
BOOST_LIB_LOCATION := /usr/lib/x86_64-linux-gnu
BOOST_LIB_FILE := boost_python3
BOOST_LIB_NUMPY := boost_numpy3

CC := gcc

CFLAGS := -std=c++17 -c -fPIC
CInc := -I$(BOOST_INC) -I$(PYTHON_INC) -I$(XTENSOR_INC)

CLinkFlags = -shared -Wl,-soname,$@ -Wl,-rpath,$(BOOST_LIB_LOCATION) -L$(BOOST_LIB_LOCATION) -l$(BOOST_LIB_FILE) -l$(BOOST_LIB_NUMPY)

PHONY: all
all: sync_motion_sim_cpp.so

sync_motion_sim_cpp.so: sync_motion_sim_cpp.o

%.so: %.o
	gcc $^ $(CLinkFlags) -o $@

%.o: %.cpp
	gcc $(CFLAGS) $(CInc) $^