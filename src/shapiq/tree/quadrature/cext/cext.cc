
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstring>
#include "quadrature_tree_shap.cc"

static PyObject *quadrature_tree_shap(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"quadrature_tree_shap", quadrature_tree_shap, METH_VARARGS,
     "Compute path-dependent Shapley interaction values with Gauss-Legendre quadrature."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cext",
    "Quadrature-based path-dependent TreeSHAP kernel.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit_cext(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    if (!module)
        return NULL;
    import_array();
    return module;
}

static PyObject *quadrature_tree_shap(PyObject *self, PyObject *args)
{
    PyObject *thresholds_obj;
    PyObject *features_obj;
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *parents_obj;
    PyObject *ancestors_obj;
    PyObject *c_acc_obj;
    PyObject *values_obj;
    int max_depth;
    int num_nodes;
    PyObject *roots_obj;
    PyObject *t_obj;
    PyObject *w_obj;
    int n_feats;
    int min_order;
    int max_order;
    PyObject *subset_keys_obj;
    PyObject *subset_starts_obj;
    PyObject *subset_counts_obj;
    PyObject *order_offsets_obj;
    PyObject *X_obj;
    PyObject *out_obj;
    const char *decision_type_cptr;
    PyObject *cat_values_obj;
    PyObject *cat_start_obj;
    PyObject *cat_size_obj;
    PyObject *children_left_default_obj;

    if (!PyArg_ParseTuple(
            args, "OOOOOOOOiiOOOiiiOOOOOOsOOOO",
            &thresholds_obj,
            &features_obj,
            &children_left_obj,
            &children_right_obj,
            &parents_obj,
            &ancestors_obj,
            &c_acc_obj,
            &values_obj,
            &max_depth,
            &num_nodes,
            &roots_obj,
            &t_obj,
            &w_obj,
            &n_feats,
            &min_order,
            &max_order,
            &subset_keys_obj,
            &subset_starts_obj,
            &subset_counts_obj,
            &order_offsets_obj,
            &X_obj,
            &out_obj,
            &decision_type_cptr,
            &cat_values_obj,
            &cat_start_obj,
            &cat_size_obj,
            &children_left_default_obj))
        return NULL;

    PyArrayObject *thresholds_array = (PyArrayObject *)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject *)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_left_array = (PyArrayObject *)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject *)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *parents_array = (PyArrayObject *)PyArray_FROM_OTF(parents_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ancestors_array = (PyArrayObject *)PyArray_FROM_OTF(ancestors_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *c_acc_array = (PyArrayObject *)PyArray_FROM_OTF(c_acc_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *roots_array = (PyArrayObject *)PyArray_FROM_OTF(roots_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *t_array = (PyArrayObject *)PyArray_FROM_OTF(t_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *w_array = (PyArrayObject *)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *subset_keys_array = (PyArrayObject *)PyArray_FROM_OTF(subset_keys_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *subset_starts_array = (PyArrayObject *)PyArray_FROM_OTF(subset_starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *subset_counts_array = (PyArrayObject *)PyArray_FROM_OTF(subset_counts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *order_offsets_array = (PyArrayObject *)PyArray_FROM_OTF(order_offsets_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_array = (PyArrayObject *)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_array = (PyArrayObject *)PyArray_FROM_OTF(out_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject *cat_values_array = (PyArrayObject *)PyArray_FROM_OTF(cat_values_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cat_start_array = (PyArrayObject *)PyArray_FROM_OTF(cat_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cat_size_array = (PyArrayObject *)PyArray_FROM_OTF(cat_size_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_left_default_array = (PyArrayObject *)PyArray_FROM_OTF(children_left_default_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);

    if (!thresholds_array || !features_array || !children_left_array || !children_right_array ||
        !parents_array || !ancestors_array || !c_acc_array || !values_array || !roots_array ||
        !t_array ||
        !w_array || !subset_keys_array || !subset_starts_array || !subset_counts_array ||
        !order_offsets_array || !X_array || !out_array || !cat_values_array ||
        !cat_start_array || !cat_size_array || !children_left_default_array)
    {
        Py_XDECREF(thresholds_array);
        Py_XDECREF(features_array);
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(parents_array);
        Py_XDECREF(ancestors_array);
        Py_XDECREF(c_acc_array);
        Py_XDECREF(values_array);
        Py_XDECREF(roots_array);
        Py_XDECREF(t_array);
        Py_XDECREF(w_array);
        Py_XDECREF(subset_keys_array);
        Py_XDECREF(subset_starts_array);
        Py_XDECREF(subset_counts_array);
        Py_XDECREF(order_offsets_array);
        Py_XDECREF(X_array);
        Py_XDECREF(cat_values_array);
        Py_XDECREF(cat_start_array);
        Py_XDECREF(cat_size_array);
        Py_XDECREF(children_left_default_array);
        if (out_array)
            PyArray_ResolveWritebackIfCopy(out_array);
        Py_XDECREF(out_array);
        return NULL;
    }

    QuadTree tree;
    tree.thresholds = (const double *)PyArray_DATA(thresholds_array);
    tree.features = (const int *)PyArray_DATA(features_array);
    tree.children_left = (const int *)PyArray_DATA(children_left_array);
    tree.children_right = (const int *)PyArray_DATA(children_right_array);
    tree.parents = (const int *)PyArray_DATA(parents_array);
    tree.ancestors = (const int *)PyArray_DATA(ancestors_array);
    tree.c_acc = (const double *)PyArray_DATA(c_acc_array);
    tree.values = (const double *)PyArray_DATA(values_array);
    tree.cat_values = (const int64_t *)PyArray_DATA(cat_values_array);
    tree.cat_start = (const int64_t *)PyArray_DATA(cat_start_array);
    tree.cat_size = (const int64_t *)PyArray_DATA(cat_size_array);
    tree.children_left_default = (const unsigned char *)PyArray_DATA(children_left_default_array);
    tree.max_depth = max_depth;
    tree.num_nodes = num_nodes;
    tree.decision_type = (std::string(decision_type_cptr) == "<") ? Q_LESS_THAN : Q_LESS_EQUAL;

    const double *t = (const double *)PyArray_DATA(t_array);
    const double *w = (const double *)PyArray_DATA(w_array);
    const double *X = (const double *)PyArray_DATA(X_array);
    double *out = (double *)PyArray_DATA(out_array);

    // validate dimensions before any PyArray_DIM(_, 1) read: on a 1-D array that would read
    // past the dims block, and a mis-sized out array would let the kernel corrupt the heap
    const char *arg_error = NULL;
    if (num_nodes < 1 || n_feats < 1 || min_order < 1 || max_order < min_order || max_depth < 0)
        arg_error = "invalid tree or order parameters.";
    else if (PyArray_NDIM(X_array) != 2 || PyArray_NDIM(out_array) != 2)
        arg_error = "X and out must be 2-dimensional arrays.";
    else if (PyArray_NDIM(roots_array) != 1 || PyArray_DIM(roots_array, 0) < 1)
        arg_error = "roots must be a 1-dimensional array with at least one entry.";
    else if (PyArray_NDIM(t_array) != 1 || PyArray_NDIM(w_array) != 1 ||
             PyArray_DIM(t_array, 0) != PyArray_DIM(w_array, 0) || PyArray_DIM(t_array, 0) < 1)
        arg_error = "quadrature nodes and weights must be 1-dimensional arrays of equal length.";
    else if (PyArray_DIM(thresholds_array, 0) != num_nodes ||
             PyArray_DIM(features_array, 0) != num_nodes ||
             PyArray_DIM(children_left_array, 0) != num_nodes ||
             PyArray_DIM(children_right_array, 0) != num_nodes ||
             PyArray_DIM(parents_array, 0) != num_nodes ||
             PyArray_DIM(ancestors_array, 0) != num_nodes ||
             PyArray_DIM(c_acc_array, 0) != num_nodes ||
             PyArray_DIM(values_array, 0) != num_nodes ||
             PyArray_DIM(cat_start_array, 0) != num_nodes ||
             PyArray_DIM(cat_size_array, 0) != num_nodes ||
             PyArray_DIM(children_left_default_array, 0) != num_nodes)
        arg_error = "per-node arrays must have length num_nodes.";
    else if (PyArray_DIM(X_array, 1) < n_feats)
        arg_error = "X must have at least n_features columns.";
    else if (PyArray_DIM(out_array, 0) != PyArray_DIM(X_array, 0))
        arg_error = "out must have one row per row of X.";
    else if (PyArray_NDIM(subset_keys_array) != 1 ||
             PyArray_NDIM(subset_starts_array) != 1 || PyArray_NDIM(subset_counts_array) != 1 ||
             PyArray_NDIM(order_offsets_array) != 1 ||
             PyArray_DIM(subset_starts_array, 0) != max_order + 1 ||
             PyArray_DIM(subset_counts_array, 0) != max_order + 1 ||
             PyArray_DIM(order_offsets_array, 0) != max_order + 1)
        arg_error = "subset table descriptors must be 1-dimensional arrays of length max_order + 1.";
    else
    {
        // the out width and every table slice must be consistent with the descriptors, or the
        // kernel would write past the buffers
        const int64_t *starts = (const int64_t *)PyArray_DATA(subset_starts_array);
        const int64_t *counts = (const int64_t *)PyArray_DATA(subset_counts_array);
        const int64_t *offsets = (const int64_t *)PyArray_DATA(order_offsets_array);
        const int64_t n_keys = (int64_t)PyArray_DIM(subset_keys_array, 0);
        const int64_t out_width = (int64_t)PyArray_DIM(out_array, 1);
        int64_t expected = (min_order <= 1) ? n_feats : 0;
        if (min_order <= 1 && offsets[1] != 0)
            arg_error = "the order-1 block must start at output position 0.";
        for (int order = std::max(min_order, 2); arg_error == NULL && order <= max_order; ++order)
        {
            if (counts[order] < 0 || starts[order] < 0 ||
                starts[order] + counts[order] * order > n_keys)
                arg_error = "subset tables are inconsistent with the subset_keys length.";
            else if (offsets[order] < 0 || offsets[order] + counts[order] > out_width)
                arg_error = "an order's output block lies outside the out array.";
            else
                expected += counts[order];
        }
        if (arg_error == NULL && out_width != expected)
            arg_error = "out width does not match the number of requested interactions.";
    }

    const int *roots = (const int *)PyArray_DATA(roots_array);
    const int n_trees = (int)PyArray_DIM(roots_array, 0);
    if (arg_error == NULL)
    {
        for (int ti = 0; ti < n_trees; ++ti)
        {
            if (roots[ti] < 0 || roots[ti] >= num_nodes)
            {
                arg_error = "roots entries must lie within [0, num_nodes).";
                break;
            }
        }
    }
    if (arg_error == NULL)
    {
        const int n_quad = (int)PyArray_DIM(t_array, 0);
        const int n_row = (int)PyArray_DIM(X_array, 0);
        const int n_col = (int)PyArray_DIM(X_array, 1);
        const int64_t out_stride = (int64_t)PyArray_DIM(out_array, 1);
        quadrature_tree_shap(tree, t, w, n_quad, roots, n_trees, n_feats, min_order, max_order,
                             (const int32_t *)PyArray_DATA(subset_keys_array),
                             (const int64_t *)PyArray_DATA(subset_starts_array),
                             (const int64_t *)PyArray_DATA(subset_counts_array),
                             (const int64_t *)PyArray_DATA(order_offsets_array),
                             X, n_row, n_col, out_stride, out);
    }

    Py_XDECREF(thresholds_array);
    Py_XDECREF(features_array);
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(parents_array);
    Py_XDECREF(ancestors_array);
    Py_XDECREF(c_acc_array);
    Py_XDECREF(values_array);
    Py_XDECREF(roots_array);
    Py_XDECREF(t_array);
    Py_XDECREF(w_array);
    Py_XDECREF(subset_keys_array);
    Py_XDECREF(subset_starts_array);
    Py_XDECREF(subset_counts_array);
    Py_XDECREF(order_offsets_array);
    Py_XDECREF(X_array);
    Py_XDECREF(cat_values_array);
    Py_XDECREF(cat_start_array);
    Py_XDECREF(cat_size_array);
    Py_XDECREF(children_left_default_array);
    PyArray_ResolveWritebackIfCopy(out_array);
    Py_XDECREF(out_array);

    if (arg_error != NULL)
    {
        PyErr_SetString(PyExc_ValueError, arg_error);
        return NULL;
    }
    Py_RETURN_NONE;
}
