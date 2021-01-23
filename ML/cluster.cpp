#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cluster.hpp"

namespace py = pybind11;
//py::module sys = py::module::import("sys");
//sys.attr("path").attr("insert")(1,CUSTOM_SYS_PATH);
PYBIND11_MODULE(cluster,m){
    //py::module m("cluster","cluster made by pybind11");
    py::class_<Cluster>(m,"Cluster")
        .def(py::init<vector<float>,int>())
        .def_readwrite("n",&Cluster::n)
        .def_readwrite("id",&Cluster::id);

    py::class_<Heap>(m,"Heap")
        .def(py::init<std::vector<std::vector<float>>>())
        .def_readwrite("n",&Heap::n)
        .def_readwrite("flag",&Heap::flag)
        .def_readwrite("heap",&Heap::heap)
        .def("shiftUp", &Heap::shiftUp)
        .def("shiftDown",&Heap::shiftDown)
        .def("remove",&Heap::remove)
        .def("calc",&Heap::calc)
        .def("judge",&Heap::judge);

}

