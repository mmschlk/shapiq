#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstring>
#include <vector>

#include "moebius.cc"

// PyObject *self is required by the Python C API but unused for module-level
// functions. See: https://docs.python.org/3/extending/extending.html
static PyObject *compute_moebius_transform(PyObject *self, PyObject *args);
static PyObject *counter_increment_py(PyObject *self, PyObject *args);
static PyObject *counter_test_bit_py(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"compute_moebius_transform", compute_moebius_transform, METH_VARARGS,
     "Compute Moebius coefficients for a set of coalitions (CSR input)."},
    {"_counter_increment", counter_increment_py, METH_VARARGS,
     "Test-only: increment a multi-word uint64 subset counter in place."},
    {"_counter_test_bit", counter_test_bit_py, METH_VARARGS,
     "Test-only: test bit b of a multi-word uint64 subset counter."},
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

// Validate a test-only counter argument: a 1-d, C-contiguous, writable uint64
// numpy array. Returned as borrowed pointer (no new reference); NULL + Python
// error set on failure. The exact-type requirement (instead of
// PyArray_FROM_OTF) keeps _counter_increment in-place: a silent conversion
// copy would swallow the mutation.
static PyArrayObject *counter_array_checked(PyObject *obj)
{
    if (!PyArray_Check(obj))
    {
        PyErr_SetString(PyExc_TypeError, "words must be a numpy array");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    if (PyArray_TYPE(arr) != NPY_UINT64 || PyArray_NDIM(arr) != 1 ||
        !PyArray_IS_C_CONTIGUOUS(arr) || !PyArray_ISWRITEABLE(arr))
    {
        PyErr_SetString(PyExc_TypeError,
                        "words must be a 1-d contiguous writable uint64 numpy array");
        return NULL;
    }
    return arr;
}

static PyObject *counter_increment_py(PyObject *self, PyObject *args)
{
    /**
     * Test-only wrapper around moebius::counter_increment.
     *
     * Increments the multi-word subset counter `words` (uint64 numpy array,
     * word i holds bits 64*i .. 64*i+63) by one, in place, propagating the
     * carry across word boundaries. Exposed so the multi-word mechanics can be
     * unit-tested from Python without enumerating 2^64 subsets.
     */
    (void)self;

    PyObject *words_obj;
    if (!PyArg_ParseTuple(args, "O", &words_obj))
        return NULL;

    PyArrayObject *words_array = counter_array_checked(words_obj);
    if (!words_array)
        return NULL;

    moebius::counter_increment(
        (uint64_t *)PyArray_DATA(words_array),
        static_cast<int>(PyArray_DIM(words_array, 0)));

    Py_RETURN_NONE;
}

static PyObject *counter_test_bit_py(PyObject *self, PyObject *args)
{
    /**
     * Test-only wrapper around moebius::counter_test_bit.
     *
     * Returns whether bit b of the multi-word subset counter `words` is set
     * (word b // 64, position b % 64).
     */
    (void)self;

    PyObject *words_obj;
    int b;
    if (!PyArg_ParseTuple(args, "Oi", &words_obj, &b))
        return NULL;

    PyArrayObject *words_array = counter_array_checked(words_obj);
    if (!words_array)
        return NULL;

    if (b < 0 || (b >> 6) >= PyArray_DIM(words_array, 0))
    {
        PyErr_SetString(PyExc_ValueError, "bit index b is out of range for words");
        return NULL;
    }

    if (moebius::counter_test_bit((const uint64_t *)PyArray_DATA(words_array), b))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}
