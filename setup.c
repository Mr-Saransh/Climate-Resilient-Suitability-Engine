// mymodule.c
#include <Python.h>

// C function exposed to Python
static PyObject* my_add(PyObject* self, PyObject* args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    return PyLong_FromLong(a + b);
}

// Method table
static PyMethodDef MyMethods[] = {
    {"add", my_add, METH_VARARGS, "Add two numbers"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "mymodule",   // Module name
    NULL,         // Module documentation
    -1,           // Size of per-interpreter state
    MyMethods
};

// Initialization function
PyMODINIT_FUNC PyInit_mymodule(void) {
    return PyModule_Create(&mymodule);
}
