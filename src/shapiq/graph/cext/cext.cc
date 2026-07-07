#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstring>
#include <vector>

#include "moebius.cc"

// PyObject *self is required by the Python C API but unused for module-level
// functions. See: https://docs.python.org/3/extending/extending.html
static PyObject *compute_moebius_transform(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"compute_moebius_transform", compute_moebius_transform, METH_VARARGS,
     "Compute Moebius coefficients for a set of coalitions (CSR input)."},
    {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cext",
    "C-extension for the GraphSHAP-IQ Moebius transform.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cext(void)
#else
PyMODINIT_FUNC initcext(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
    if (!module)
        return NULL;
#else
    PyObject *module = Py_InitModule("cext", module_methods);
    if (!module)
        return;
#endif

    import_array(); // initialize the NumPy C API

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

static PyObject *compute_moebius_transform(PyObject *self, PyObject *args)
{
    /**
     * Compute Moebius coefficients for a set of coalitions (CSR input).
     *
     * The funcion has the following input parameters (in exact order):
     * - members_flat: int32 numpy array, concatenated members of the coalitions.
     * - offsets: int32 numpy array, CSR offsets, length n_coalitions + 1.
     * - lookup_members_flat: int32 numpy array, concatenated members of the lookup keys.
     * - lookup_offsets: int32 numpy array, CSR offsets, length n_lookup + 1.
     * - lookup_indices: int32 numpy array, prediction row index per lookup key.
     * - predictions: float64 numpy array, coalition_predictions.
     *
     * Returns a float64 numpy array of length n_coalitions containing the Moebius
     * coefficient for every coalition, in the same order as the members_flat/offsets input.
     */
    (void)self;

    PyObject *members_flat_obj;
    PyObject *offsets_obj;
    PyObject *lookup_members_flat_obj;
    PyObject *lookup_offsets_obj;
    PyObject *lookup_indices_obj;
    PyObject *predictions_obj;

    if (!PyArg_ParseTuple(args, "OOOOOO",
                          &members_flat_obj,
                          &offsets_obj,
                          &lookup_members_flat_obj,
                          &lookup_offsets_obj,
                          &lookup_indices_obj,
                          &predictions_obj))
    {
        return NULL;
    }

    PyArrayObject *members_flat_array = (PyArrayObject *)PyArray_FROM_OTF(members_flat_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *offsets_array = (PyArrayObject *)PyArray_FROM_OTF(offsets_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *lookup_members_flat_array = (PyArrayObject *)PyArray_FROM_OTF(lookup_members_flat_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *lookup_offsets_array = (PyArrayObject *)PyArray_FROM_OTF(lookup_offsets_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *lookup_indices_array = (PyArrayObject *)PyArray_FROM_OTF(lookup_indices_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *predictions_array = (PyArrayObject *)PyArray_FROM_OTF(predictions_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    if (!members_flat_array || !offsets_array || !lookup_members_flat_array ||
        !lookup_offsets_array || !lookup_indices_array || !predictions_array)
    {
        Py_XDECREF(members_flat_array);
        Py_XDECREF(offsets_array);
        Py_XDECREF(lookup_members_flat_array);
        Py_XDECREF(lookup_offsets_array);
        Py_XDECREF(lookup_indices_array);
        Py_XDECREF(predictions_array);
        PyErr_SetString(PyExc_TypeError, "All inputs must be convertible to numpy arrays");
        return NULL;
    }

    // n_coalitions = len(offsets) - 1, n_lookup = len(lookup_offsets) - 1
    const int n_coalitions = static_cast<int>(PyArray_DIM(offsets_array, 0)) - 1;
    const int n_lookup = static_cast<int>(PyArray_DIM(lookup_offsets_array, 0)) - 1;

    if (n_coalitions < 0 || n_lookup < 0)
    {
        Py_XDECREF(members_flat_array);
        Py_XDECREF(offsets_array);
        Py_XDECREF(lookup_members_flat_array);
        Py_XDECREF(lookup_offsets_array);
        Py_XDECREF(lookup_indices_array);
        Py_XDECREF(predictions_array);
        PyErr_SetString(PyExc_ValueError, "offsets arrays must have length >= 1");
        return NULL;
    }

    const int *members_flat = (const int *)PyArray_DATA(members_flat_array);
    const int *offsets = (const int *)PyArray_DATA(offsets_array);
    const int *lookup_members_flat = (const int *)PyArray_DATA(lookup_members_flat_array);
    const int *lookup_offsets = (const int *)PyArray_DATA(lookup_offsets_array);
    const int *lookup_indices = (const int *)PyArray_DATA(lookup_indices_array);
    const double *predictions = (const double *)PyArray_DATA(predictions_array);

    npy_intp dims[1] = {(npy_intp)n_coalitions};
    PyObject *output = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!output)
    {
        Py_XDECREF(members_flat_array);
        Py_XDECREF(offsets_array);
        Py_XDECREF(lookup_members_flat_array);
        Py_XDECREF(lookup_offsets_array);
        Py_XDECREF(lookup_indices_array);
        Py_XDECREF(predictions_array);
        return NULL;
    }
    double *out = (double *)PyArray_DATA((PyArrayObject *)output);

    const int missing = moebius::compute(
        members_flat, offsets, n_coalitions,
        lookup_members_flat, lookup_offsets, lookup_indices, n_lookup,
        predictions, out);

    Py_XDECREF(members_flat_array);
    Py_XDECREF(offsets_array);
    Py_XDECREF(lookup_members_flat_array);
    Py_XDECREF(lookup_offsets_array);
    Py_XDECREF(lookup_indices_array);
    Py_XDECREF(predictions_array);

    if (missing > 0)
    {
        Py_DECREF(output);
        PyErr_SetString(PyExc_ValueError,
                        "A coalition subset was missing from coalition_lookup");
        return NULL;
    }

    return output;
}
