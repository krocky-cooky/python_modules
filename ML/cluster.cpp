#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cluster.hpp"

namespace py = pybind11;
//py::module sys = py::module::import("sys");
//sys.attr("path").attr("insert")(1,CUSTOM_SYS_PATH);
PYBIND11_MODULE(cluster,m){
    //py::module m("cluster","cluster made by pybind11")

    py::class_<Heap>(m,"Heap")
        .def(py::init<std::vector<std::vector<float>>>())
        .def_readwrite("n",&Heap::n)
        .def_readwrite("debug",&Heap::debug)
        .def_readwrite("debug2",&Heap::debug2)
        .def_readwrite("debug3",&Heap::debug3)
        .def_readwrite("heap",&Heap::heap)
        .def("shiftUp", &Heap::shiftUp)
        .def("shiftDown",&Heap::shiftDown)
        .def("add",&Heap::add)
        .def("update",&Heap::update)
        .def("judge",&Heap::judge);

}

