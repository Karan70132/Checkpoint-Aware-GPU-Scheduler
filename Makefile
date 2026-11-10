CXX ?= g++
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17

.PHONY: all test clean

all: scheduler

scheduler: src/scheduler.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test:
	python3 -m pytest -q

clean:
	$(RM) scheduler
