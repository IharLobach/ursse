CC := gcc
CFLAGS := -std=c++17 -c -fPIC

PHONY: all
all: hydra_harp_reader

hydra_harp_reader: hydra_harp_reader.o


%: %.o
	gcc $^ -o $@

%.o: %.cpp
	gcc $(CFLAGS) $^