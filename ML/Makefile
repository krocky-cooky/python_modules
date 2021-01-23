

pybind: cluster.cpp cluster.hpp
	clang++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` -undefined dynamic_lookup cluster.cpp -o cluster`python3-config --extension-suffix`

sub: cluster.cpp cluster.hpp
	clang++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` cluster.cpp -o cluster`python-config --extension-suffix`