PYTHON_VERSION := 3.9
PYTHON_INC := /opt/homebrew/Caskroom/miniforge/base/envs/env/include/python$(PYTHON_VERSION)
BOOST_INC := /opt/homebrew/include/boost
BOOST_LIB_LOCATION := /opt/homebrew/lib
BOOST_LIB_FILE := boost_python39
BOOST_LIB_NUMPY := boost_numpy39

CC := gcc

CFLAGS := -std=c++17 -c -fPIC
CInc := -I$(BOOST_INC) -I$(PYTHON_INC)

CLinkFlags = -shared -Wl,-install_name,$@ -Wl,-rpath,$(BOOST_LIB_LOCATION) -L$(BOOST_LIB_LOCATION) -l$(BOOST_LIB_FILE) -l$(BOOST_LIB_NUMPY)

PHONY: all
all: sync_motion_sim_cpp.so

sync_motion_sim_cpp.so: sync_motion_sim_cpp.o

%.so: %.o
	gcc $^ $(CLinkFlags) -o $@

%.o: %.cpp
	gcc $(CFLAGS) $(CInc) $^