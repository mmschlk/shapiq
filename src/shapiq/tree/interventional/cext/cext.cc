#include <Python.h>
#include <numpy/arrayobject.h>
#include "interventional.cpp"
#include <cstring>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include <chrono>

#ifdef _MSC_VER
#define __restrict__ __restrict
#endif

using namespace std;
// PyObject *self is not used in this function, but it is required by the Python C API for defining module methods.
// It represents the module object itself when the function is called as a method of a module, but since we are defining a standalone function, we can ignore it in our implementation.
// See: https://docs.python.org/3/extending/extending.html
static PyObject *compute_interactions_batched_sparse(PyObject *self, PyObject *args);
static PyObject *compute_interactions_flatten(PyObject *self, PyObject *args);
static PyObject *preprocess_boolean_trees(PyObject *self, PyObject *args);
static PyObject *preprocess_trees_point(PyObject *self, PyObject *args);
static PyObject *compute_interactions_cohort(PyObject *self, PyObject *args);
static PyObject *predict_ensemble_sum(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"compute_interactions_batched_sparse", compute_interactions_batched_sparse, METH_VARARGS, "Compute sparse feature interactions in batches using the interventional algorithm."},
    {"compute_interactions_flatten", compute_interactions_flatten, METH_VARARGS, "Compute feature interactions with flattened input."},
    {"preprocess_boolean_trees", preprocess_boolean_trees, METH_VARARGS, "Preprocess boolean trees: DFS traversal to produce flattened E/R arrays."},
    {"preprocess_trees_point", preprocess_trees_point, METH_VARARGS, "Preprocess trees for one explain point against every reference sample: DFS traversal to produce flattened E/R arrays."},
    {"compute_interactions_cohort", compute_interactions_cohort, METH_VARARGS, "Fused dense kernel: cohort DFS sharing one tree walk across all reference samples, updating interactions at each leaf."},
    {"predict_ensemble_sum", predict_ensemble_sum, METH_VARARGS, "Route every row of X through every tree and return the per-row sum of leaf predictions."},
    {NULL, NULL, 0, NULL}};
/** Define the Python Module for both Python 3 and Python 2 Version.
 * This code is mostly copied from https://github.com/yupbank/linear_tree_shap/blob/main/linear_tree_shap/cext/_cext.cc
 */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cext",
    "This module provides an interface for computing feature interactions using the interventional algorithm.",
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
PyMODINIT_FUNC init_cext(void)
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

    /* Load `numpy` functionality. */
    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

static bool parse_index_type(const std::string &index, IndexType &index_type)
{
    /**
     * This function takes a string representation of the index type and maps it to the corresponding IndexType enum value.
     * It returns true if the mapping is successful and false if the input string does not match any supported index type.
     */
    if (index == "SII" || index == "SV")
    {
        index_type = IndexType::SII;
        return true;
    }
    if (index == "BII" || index == "BV")
    {
        index_type = IndexType::BII;
        return true;
    }
    if (index == "CHII" || index == "CV")
    {
        index_type = IndexType::CHII;
        return true;
    }
    if (index == "FBII")
    {
        index_type = IndexType::FBII;
        return true;
    }
    if (index == "FSII")
    {
        index_type = IndexType::FSII;
        return true;
    }
    if (index == "STII")
    {
        index_type = IndexType::STII;
        return true;
    }
    if (index == "Moebius")
    {
        index_type = IndexType::MOEBIUS;
        return true;
    }
    if (index == "CUSTOM")
    {
        index_type = IndexType::CUSTOM;
        return true;
    }
    return false;
}

static PyObject *bitset_to_pytuple(const BitSet &bitset)
{
    const uint64_t size = bitset.num_bits();
    PyObject *tuple = PyTuple_New(static_cast<Py_ssize_t>(size));
    if (!tuple)
    {
        return NULL;
    }

    if (size == 0)
    {
        return tuple;
    }

    std::vector<uint64_t> buffer(size);
    bitset.fill_buffer(buffer.data());
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(size); ++i)
    {
        PyObject *feature = PyLong_FromLongLong(static_cast<long long>(buffer[static_cast<size_t>(i)]));
        if (!feature)
        {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, i, feature);
    }
    return tuple;
}

static PyObject *sparse_map_to_pydict(const algorithms::SparseInteractionMap &sparse_result)
{
    PyObject *output = PyDict_New();
    if (!output)
    {
        return NULL;
    }

    for (const auto &entry : sparse_result)
    {
        PyObject *key = bitset_to_pytuple(entry.first);
        if (!key)
        {
            Py_DECREF(output);
            return NULL;
        }
        PyObject *value = PyFloat_FromDouble(entry.second);
        if (!value)
        {
            Py_DECREF(key);
            Py_DECREF(output);
            return NULL;
        }
        if (PyDict_SetItem(output, key, value) < 0)
        {
            Py_DECREF(key);
            Py_DECREF(value);
            Py_DECREF(output);
            return NULL;
        }
        Py_DECREF(key);
        Py_DECREF(value);
    }

    return output;
}

static void decref_cat_array_triples(std::vector<std::tuple<PyArrayObject *, PyArrayObject *, PyArrayObject *>> &triples)
{
    for (auto &triple : triples)
    {
        Py_XDECREF(std::get<0>(triple));
        Py_XDECREF(std::get<1>(triple));
        Py_XDECREF(std::get<2>(triple));
    }
    triples.clear();
}

static PyObject *compute_interactions_batched_sparse(PyObject *self, PyObject *args)
{
    /**
     * Computes interactions for each tree and returns a sparse representation as a dictionary mapping feature subsets to interaction values.
     * The input parameters are the same as the batched version, but the output is different to accommodate the sparse representation.
     * This function should be used when the max_order exceeds 2.
     * The funcion has the following input parameters( in exact order):
     * - leaf_predictions: A list of numpy arrays, where each array contains the predictions at the leaf nodes of a decision tree.
     * - thresholds: A list of numpy arrays, where each array contains the threshold values for the decision nodes in a tree.
     * - features: A list of numpy arrays, where each array contains the feature indices used for splitting at each decision node in a tree.
     * - children_left: A list of numpy arrays, where each array contains the indices of the left child nodes for each decision node in a tree.
     * - children_right: A list of numpy arrays, where each array contains the indices of the right child nodes for each decision node in a tree.
     * - children_missing: A list of boolean numpy arrays, where each array indicates whether the left node should be followed when the feature value is missing (NaN) for each decision node in a tree.
     * - reference_data: A numpy array containing the reference data samples used for computing interactions.
     * - explain_data: A numpy array containing the data sample for which interactions are to be computed.
     * - decision_type: A string indicating the type of decision tree (e.g., "classification" or "regression").
     * - index: A string indicating the type of interaction index to compute (e.g., "shapley" or "banzhaf").
     * - max_order: An integer specifying the maximum order of interactions to compute.
     * - verbose: An integer specifying the verbosity level for logging during computation.
     * The function returns a dictionary where the keys are frozensets of feature indices representing the subsets of features involved in the interactions, and the values are the corresponding interaction values computed based on the provided input parameters and the interventional algorithm.
     */
    // Tree parameters
    PyObject *leaf_predictions_obj;
    PyObject *thresholds_obj;
    PyObject *features_obj;
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *children_missing_obj;
    const char *decision_type_cptr;

    // Algorithm parameters
    PyObject *reference_data_obj;
    PyObject *explain_data_obj;
    const char *index_cptr;
    int max_order;
    int verbose;

    // Optional custom weight table (None when not used)
    PyObject *weight_table_obj = Py_None;
    // Optional categorical split CSR lists (None for numeric-only ensembles): per tree,
    // int64 arrays cat_values / cat_start / cat_size mirroring TreeModel's layout
    PyObject *cat_values_obj = Py_None;
    PyObject *cat_start_obj = Py_None;
    PyObject *cat_size_obj = Py_None;

    if (!PyArg_ParseTuple(args, "OOOOOOOOssii|OOOO", &leaf_predictions_obj, &thresholds_obj, &features_obj, &children_left_obj, &children_right_obj, &children_missing_obj, &reference_data_obj, &explain_data_obj, &decision_type_cptr, &index_cptr, &max_order, &verbose, &weight_table_obj, &cat_values_obj, &cat_start_obj, &cat_size_obj))
    {
        return NULL;
    }

    const bool use_categorical = (cat_values_obj != Py_None);
    if (use_categorical && (!PyList_Check(cat_values_obj) || !PyList_Check(cat_start_obj) || !PyList_Check(cat_size_obj)))
    {
        PyErr_SetString(PyExc_TypeError, "cat_values, cat_start, and cat_size must be lists of numpy arrays (or None)");
        return NULL;
    }

    if (!PyList_Check(leaf_predictions_obj) || !PyList_Check(thresholds_obj) || !PyList_Check(features_obj) || !PyList_Check(children_left_obj) || !PyList_Check(children_right_obj) || !PyList_Check(children_missing_obj) || !PyArray_Check(reference_data_obj) || !PyArray_Check(explain_data_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Input data must be lists of tree arrays and numpy arrays for reference/explain data");
        return NULL;
    }

    if (max_order < 1)
    {
        PyErr_SetString(PyExc_ValueError, "max_order must be >= 1");
        return NULL;
    }

    PyObject *iterator_leaf = PyObject_GetIter(leaf_predictions_obj);
    PyObject *iterator_thresholds = PyObject_GetIter(thresholds_obj);
    PyObject *iterator_features = PyObject_GetIter(features_obj);
    PyObject *iterator_children_left = PyObject_GetIter(children_left_obj);
    PyObject *iterator_children_right = PyObject_GetIter(children_right_obj);
    PyObject *iterator_children_missing = PyObject_GetIter(children_missing_obj);
    std::string index = std::string(index_cptr);

    if (!iterator_leaf || !iterator_thresholds || !iterator_features || !iterator_children_left || !iterator_children_right || !iterator_children_missing)
    {
        Py_XDECREF(iterator_leaf);
        Py_XDECREF(iterator_thresholds);
        Py_XDECREF(iterator_features);
        Py_XDECREF(iterator_children_left);
        Py_XDECREF(iterator_children_right);
        Py_XDECREF(iterator_children_missing);
        PyErr_SetString(PyExc_TypeError, "Input data must be lists of numpy arrays");
        return NULL;
    }

    PyArrayObject *leaf_predictions_array, *thresholds_array, *features_array, *children_left_array, *children_right_array, *children_missing_array;
    PyObject *leaf_pred_iter, *thresholds_iter, *features_iter, *children_left_iter, *children_right_iter, *children_missing_iter;
    std::vector<Tree> trees;
    std::vector<std::tuple<PyArrayObject *, PyArrayObject *, PyArrayObject *, PyArrayObject *, PyArrayObject *, PyArrayObject *>> arrays_for_decref;
    std::vector<std::tuple<PyArrayObject *, PyArrayObject *, PyArrayObject *>> cat_arrays_for_decref;
    int num_trees = 0;

    while (1)
    {
        leaf_pred_iter = PyIter_Next(iterator_leaf);
        thresholds_iter = PyIter_Next(iterator_thresholds);
        features_iter = PyIter_Next(iterator_features);
        children_left_iter = PyIter_Next(iterator_children_left);
        children_right_iter = PyIter_Next(iterator_children_right);
        children_missing_iter = PyIter_Next(iterator_children_missing);

        if (!leaf_pred_iter || !thresholds_iter || !features_iter || !children_left_iter || !children_right_iter || !children_missing_iter)
        {
            if (leaf_pred_iter || thresholds_iter || features_iter || children_left_iter || children_right_iter || children_missing_iter)
            {
                Py_XDECREF(leaf_pred_iter);
                Py_XDECREF(thresholds_iter);
                Py_XDECREF(features_iter);
                Py_XDECREF(children_left_iter);
                Py_XDECREF(children_right_iter);
                Py_XDECREF(children_missing_iter);
                Py_XDECREF(iterator_leaf);
                Py_XDECREF(iterator_thresholds);
                Py_XDECREF(iterator_features);
                Py_XDECREF(iterator_children_left);
                Py_XDECREF(iterator_children_right);
                Py_XDECREF(iterator_children_missing);
                for (auto &arrays_tuple : arrays_for_decref)
                {
                    Py_XDECREF(std::get<0>(arrays_tuple));
                    Py_XDECREF(std::get<1>(arrays_tuple));
                    Py_XDECREF(std::get<2>(arrays_tuple));
                    Py_XDECREF(std::get<3>(arrays_tuple));
                    Py_XDECREF(std::get<4>(arrays_tuple));
                    Py_XDECREF(std::get<5>(arrays_tuple));
                }
                decref_cat_array_triples(cat_arrays_for_decref);
                PyErr_SetString(PyExc_ValueError, "Input lists of numpy arrays must be of the same length");
                return NULL;
            }
            break;
        }

        leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_pred_iter, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        thresholds_array = (PyArrayObject *)PyArray_FROM_OTF(thresholds_iter, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        features_array = (PyArrayObject *)PyArray_FROM_OTF(features_iter, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        children_left_array = (PyArrayObject *)PyArray_FROM_OTF(children_left_iter, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        children_right_array = (PyArrayObject *)PyArray_FROM_OTF(children_right_iter, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        children_missing_array = (PyArrayObject *)PyArray_FROM_OTF(children_missing_iter, NPY_BOOL, NPY_ARRAY_IN_ARRAY);

        Py_XDECREF(leaf_pred_iter);
        Py_XDECREF(thresholds_iter);
        Py_XDECREF(features_iter);
        Py_XDECREF(children_left_iter);
        Py_XDECREF(children_right_iter);
        Py_XDECREF(children_missing_iter);

        if (!leaf_predictions_array || !thresholds_array || !features_array || !children_left_array || !children_right_array || !children_missing_array)
        {
            Py_XDECREF(leaf_predictions_array);
            Py_XDECREF(thresholds_array);
            Py_XDECREF(features_array);
            Py_XDECREF(children_left_array);
            Py_XDECREF(children_right_array);
            Py_XDECREF(children_missing_array);
            for (auto &arrays_tuple : arrays_for_decref)
            {
                Py_XDECREF(std::get<0>(arrays_tuple));
                Py_XDECREF(std::get<1>(arrays_tuple));
                Py_XDECREF(std::get<2>(arrays_tuple));
                Py_XDECREF(std::get<3>(arrays_tuple));
                Py_XDECREF(std::get<4>(arrays_tuple));
                Py_XDECREF(std::get<5>(arrays_tuple));
            }
            Py_XDECREF(iterator_leaf);
            Py_XDECREF(iterator_thresholds);
            Py_XDECREF(iterator_features);
            Py_XDECREF(iterator_children_left);
            Py_XDECREF(iterator_children_right);
            Py_XDECREF(iterator_children_missing);
            decref_cat_array_triples(cat_arrays_for_decref);
            PyErr_SetString(PyExc_TypeError, "Each tree's parameters must be numpy arrays");
            return NULL;
        }

        const int64_t *cat_values_ptr = nullptr;
        const int64_t *cat_start_ptr = nullptr;
        const int64_t *cat_size_ptr = nullptr;
        if (use_categorical)
        {
            // borrowed references; conversion below creates owned arrays tracked for decref
            PyObject *cat_values_item = PyList_GetItem(cat_values_obj, num_trees);
            PyObject *cat_start_item = PyList_GetItem(cat_start_obj, num_trees);
            PyObject *cat_size_item = PyList_GetItem(cat_size_obj, num_trees);
            PyArrayObject *cat_values_array = cat_values_item ? (PyArrayObject *)PyArray_FROM_OTF(cat_values_item, NPY_INT64, NPY_ARRAY_IN_ARRAY) : NULL;
            PyArrayObject *cat_start_array = cat_start_item ? (PyArrayObject *)PyArray_FROM_OTF(cat_start_item, NPY_INT64, NPY_ARRAY_IN_ARRAY) : NULL;
            PyArrayObject *cat_size_array = cat_size_item ? (PyArrayObject *)PyArray_FROM_OTF(cat_size_item, NPY_INT64, NPY_ARRAY_IN_ARRAY) : NULL;
            if (!cat_values_array || !cat_start_array || !cat_size_array)
            {
                Py_XDECREF(cat_values_array);
                Py_XDECREF(cat_start_array);
                Py_XDECREF(cat_size_array);
                Py_XDECREF(leaf_predictions_array);
                Py_XDECREF(thresholds_array);
                Py_XDECREF(features_array);
                Py_XDECREF(children_left_array);
                Py_XDECREF(children_right_array);
                Py_XDECREF(children_missing_array);
                for (auto &arrays_tuple : arrays_for_decref)
                {
                    Py_XDECREF(std::get<0>(arrays_tuple));
                    Py_XDECREF(std::get<1>(arrays_tuple));
                    Py_XDECREF(std::get<2>(arrays_tuple));
                    Py_XDECREF(std::get<3>(arrays_tuple));
                    Py_XDECREF(std::get<4>(arrays_tuple));
                    Py_XDECREF(std::get<5>(arrays_tuple));
                }
                decref_cat_array_triples(cat_arrays_for_decref);
                Py_XDECREF(iterator_leaf);
                Py_XDECREF(iterator_thresholds);
                Py_XDECREF(iterator_features);
                Py_XDECREF(iterator_children_left);
                Py_XDECREF(iterator_children_right);
                Py_XDECREF(iterator_children_missing);
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_TypeError, "Each tree's categorical arrays must be int64 numpy arrays");
                return NULL;
            }
            cat_arrays_for_decref.push_back(std::make_tuple(cat_values_array, cat_start_array, cat_size_array));
            cat_values_ptr = (const int64_t *)PyArray_DATA(cat_values_array);
            cat_start_ptr = (const int64_t *)PyArray_DATA(cat_start_array);
            cat_size_ptr = (const int64_t *)PyArray_DATA(cat_size_array);
        }

        trees.push_back(Tree(
            (double *)PyArray_DATA(leaf_predictions_array),
            (double *)PyArray_DATA(thresholds_array),
            (int64_t *)PyArray_DATA(features_array),
            (int64_t *)PyArray_DATA(children_left_array),
            (int64_t *)PyArray_DATA(children_right_array),
            (bool *)PyArray_DATA(children_missing_array),
            std::string(decision_type_cptr),
            cat_values_ptr,
            cat_start_ptr,
            cat_size_ptr));
        arrays_for_decref.push_back(std::make_tuple(leaf_predictions_array, thresholds_array, features_array, children_left_array, children_right_array, children_missing_array));
        num_trees++;
    }

    Py_DECREF(iterator_leaf);
    Py_DECREF(iterator_thresholds);
    Py_DECREF(iterator_features);
    Py_DECREF(iterator_children_left);
    Py_DECREF(iterator_children_right);
    Py_DECREF(iterator_children_missing);

    if (num_trees == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Input lists of numpy arrays must not be empty");
        return NULL;
    }

    PyArrayObject *reference_data_array = (PyArrayObject *)PyArray_FROM_OTF(reference_data_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *explain_data_array = (PyArrayObject *)PyArray_FROM_OTF(explain_data_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!reference_data_array || !explain_data_array)
    {
        Py_XDECREF(reference_data_array);
        Py_XDECREF(explain_data_array);
        for (const auto &arr_tuple : arrays_for_decref)
        {
            Py_XDECREF(std::get<0>(arr_tuple));
            Py_XDECREF(std::get<1>(arr_tuple));
            Py_XDECREF(std::get<2>(arr_tuple));
            Py_XDECREF(std::get<3>(arr_tuple));
            Py_XDECREF(std::get<4>(arr_tuple));
            Py_XDECREF(std::get<5>(arr_tuple));
        }
        decref_cat_array_triples(cat_arrays_for_decref);
        PyErr_SetString(PyExc_TypeError, "Reference and explain data must be numpy arrays");
        return NULL;
    }

    double *reference_data = (double *)PyArray_DATA(reference_data_array);
    double *explain_data = (double *)PyArray_DATA(explain_data_array);
    int n_reference_samples = static_cast<int>(reference_data_array->dimensions[0]);
    int n_features = static_cast<int>(reference_data_array->dimensions[1]);

    IndexType index_type;
    if (!parse_index_type(index, index_type))
    {
        Py_XDECREF(reference_data_array);
        Py_XDECREF(explain_data_array);
        for (const auto &arr_tuple : arrays_for_decref)
        {
            Py_XDECREF(std::get<0>(arr_tuple));
            Py_XDECREF(std::get<1>(arr_tuple));
            Py_XDECREF(std::get<2>(arr_tuple));
            Py_XDECREF(std::get<3>(arr_tuple));
            Py_XDECREF(std::get<4>(arr_tuple));
            Py_XDECREF(std::get<5>(arr_tuple));
        }
        decref_cat_array_triples(cat_arrays_for_decref);
        PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + index).c_str());
        return NULL;
    }

    // Extract custom weight table pointer if provided
    const double *custom_table = nullptr;
    int64_t custom_N = 0, custom_K = 0;
    PyArrayObject *weight_table_array = nullptr;
    if (weight_table_obj != Py_None)
    {
        weight_table_array = (PyArrayObject *)PyArray_FROM_OTF(weight_table_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (!weight_table_array)
        {
            Py_XDECREF(reference_data_array);
            Py_XDECREF(explain_data_array);
            for (const auto &arr_tuple : arrays_for_decref)
            {
                Py_XDECREF(std::get<0>(arr_tuple));
                Py_XDECREF(std::get<1>(arr_tuple));
                Py_XDECREF(std::get<2>(arr_tuple));
                Py_XDECREF(std::get<3>(arr_tuple));
                Py_XDECREF(std::get<4>(arr_tuple));
                Py_XDECREF(std::get<5>(arr_tuple));
            }
            decref_cat_array_triples(cat_arrays_for_decref);
            PyErr_SetString(PyExc_TypeError, "weight_table must be a float64 numpy array");
            return NULL;
        }
        custom_table = (const double *)PyArray_DATA(weight_table_array);
        custom_N = (int64_t)n_features + 1;
        custom_K = (int64_t)max_order + 1;
    }

    algorithms::SparseInteractionMap sparse_result;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel
    {
        algorithms::SparseInteractionMap local_sparse_result;
        inter_weights::WeightCache weight_cache = (custom_table != nullptr)
                                                      ? inter_weights::WeightCache((uint64_t)(2 * n_features), custom_table, custom_N, custom_K)
                                                      : inter_weights::WeightCache((uint64_t)(2 * n_features));

#pragma omp for nowait
        for (int t = 0; t < num_trees; t++)
        {
            for (int i = 0; i < n_reference_samples; i++)
            {
                double *reference_sample = reference_data + i * n_features;
                algorithms::compute_interactions_sparse(
                    trees[t],
                    local_sparse_result,
                    weight_cache,
                    reference_sample,
                    explain_data,
                    n_features,
                    index_type,
                    max_order,
                    verbose);
            }
        }

#pragma omp critical
        {
            for (const auto &entry : local_sparse_result)
            {
                sparse_result[entry.first] += entry.second;
            }
        }
    }
    Py_END_ALLOW_THREADS

    // Keep behavior aligned with existing batched method: average over reference samples only.
    for (auto &entry : sparse_result)
    {
        entry.second /= static_cast<double>(n_reference_samples);
    }

    PyObject *output = sparse_map_to_pydict(sparse_result);

    Py_XDECREF(reference_data_array);
    Py_XDECREF(explain_data_array);
    Py_XDECREF(weight_table_array);
    for (const auto &arr_tuple : arrays_for_decref)
    {
        Py_XDECREF(std::get<0>(arr_tuple));
        Py_XDECREF(std::get<1>(arr_tuple));
        Py_XDECREF(std::get<2>(arr_tuple));
        Py_XDECREF(std::get<3>(arr_tuple));
        Py_XDECREF(std::get<4>(arr_tuple));
        Py_XDECREF(std::get<5>(arr_tuple));
    }
    decref_cat_array_triples(cat_arrays_for_decref);

    return output;
}

// === Optimized helpers for compute_interactions_flatten ===

// Precompute weight lookup tables. The signed weights come from inter_weights::weight_func.
// table_s1: indexed by [s_cap_e * stride^2 + e * stride + r], s_cap_e in {0,1}
// table_s2: indexed by [s_cap_e_combined * stride^2 + e * stride + r], s_cap_e_combined in {0,1,2}
// table_s3: indexed by [s_cap_e_combined * stride^2 + e * stride + r], s_cap_e_combined in {0,1,2,3}
static void precompute_weight_tables(
    IndexType index_type, int n_features, int max_order,
    double *table_s1, double *table_s2, double *table_s3, int table_stride)
{
    // table_stride should be max(max_e, max_r) + 1, not n_features + 1.
    // This avoids O(n^3) precomputation when n_features is large but actual e/r values are small.
    int max_val = table_stride - 1;
    for (int s_cap_e = 0; s_cap_e <= 1; s_cap_e++)
    {
        int s_cap_r = 1 - s_cap_e;
        for (int e = 0; e <= max_val; e++)
        {
            for (int r = 0; r <= max_val; r++)
            {
                double w = inter_weights::weight_func(
                    n_features, e, r, s_cap_e, s_cap_r, 1, index_type, max_order);
                table_s1[s_cap_e * table_stride * table_stride + e * table_stride + r] = (double)w;
            }
        }
    }
    if (max_order >= 2 && table_s2)
    {
        for (int s_cap_e_c = 0; s_cap_e_c <= 2; s_cap_e_c++)
        {
            int s_cap_r_c = 2 - s_cap_e_c;
            for (int e = 0; e <= max_val; e++)
            {
                for (int r = 0; r <= max_val; r++)
                {
                    double w = inter_weights::weight_func(
                        n_features, e, r, s_cap_e_c, s_cap_r_c, 2, index_type, max_order);
                    table_s2[s_cap_e_c * table_stride * table_stride + e * table_stride + r] = (double)w;
                }
            }
        }
    }
    if (max_order >= 3 && table_s3)
    {
        for (int s_cap_e_c = 0; s_cap_e_c <= 3; s_cap_e_c++)
        {
            int s_cap_r_c = 3 - s_cap_e_c;
            for (int e = 0; e <= max_val; e++)
            {
                for (int r = 0; r <= max_val; r++)
                {
                    double w = inter_weights::weight_func(
                        n_features, e, r, s_cap_e_c, s_cap_r_c, 3, index_type, max_order);
                    table_s3[s_cap_e_c * table_stride * table_stride + e * table_stride + r] = (double)w;
                }
            }
        }
    }
}

// Compact index for order-3 triple (i < j < k):
//   base = n + n*(n-1)/2
//   offset = i + j*(j-1)/2 + k*(k-1)*(k-2)/6
static inline int index3(int i, int j, int k, int n)
{
    // Ensure i < j < k
    if (i > j) { int t = i; i = j; j = t; }
    if (j > k) { int t = j; j = k; k = t; }
    if (i > j) { int t = i; i = j; j = t; }
    int base = n + n * (n - 1) / 2;
    return base + i + j * (j - 1) / 2 + k * (k - 1) * (k - 2) / 6;
}

// Template: order-1 two-pass computation (vectorized multiply + OpenMP scatter)
template <IndexType IT>
static void compute_order1_twopass(
    const double *__restrict__ leaf_pred,
    const int32_t *__restrict__ feat32,
    const int32_t *__restrict__ e32,
    const int32_t *__restrict__ r32,
    const int32_t *__restrict__ fie32,
    const double *__restrict__ e_vals,
    const double *__restrict__ r_vals,
    const double *__restrict__ fie_vals,
    int n_iterations,
    int n_features,
    double inv_scaling,
    const double *__restrict__ table_s1,
    int table_stride,
    inter_weights::WeightCache &weight_cache,
    int max_order,
    double *__restrict__ result)
{
    double *contrib = new double[n_iterations];

    if constexpr (IT == IndexType::BII)
    {
        for (int i = 0; i < n_iterations; i++)
        {
            double sign = 1.0 - 2.0 * (1.0 - fie_vals[i]);
            double w = exp2(-(e_vals[i] + r_vals[i] - 1.0));
            contrib[i] = leaf_pred[i] * sign * w * inv_scaling;
        }
    }
    else if constexpr (IT == IndexType::CUSTOM)
    {
        for (int i = 0; i < n_iterations; i++)
        {
            double w = (double)weight_cache.get_weight(
                n_features, e32[i], r32[i], fie32[i], 1 - fie32[i], 1, IT, max_order);
            contrib[i] = leaf_pred[i] * w * inv_scaling;
        }
    }
    else
    {
        double *weights = new double[n_iterations];
        for (int i = 0; i < n_iterations; i++)
        {
            weights[i] = table_s1[fie32[i] * table_stride * table_stride + e32[i] * table_stride + r32[i]];
        }
        for (int i = 0; i < n_iterations; i++)
        {
            contrib[i] = leaf_pred[i] * weights[i] * inv_scaling;
        }
        delete[] weights;
    }

    // Pass 2: parallel scatter-add with thread-local result arrays
    #pragma omp parallel
    {
        double *local_result = new double[n_features]();
        #pragma omp for schedule(static)
        for (int i = 0; i < n_iterations; i++)
        {
            local_result[feat32[i]] += (double)contrib[i];
        }
        #pragma omp critical
        {
            for (int k = 0; k < n_features; k++)
                result[k] += local_result[k];
        }
        delete[] local_result;
    }

    delete[] contrib;
}

// Template: order-2 computation parallelized over leaves
template <IndexType IT>
static void compute_order2_leafparallel(
    const double *__restrict__ leaf_pred,
    const int32_t *__restrict__ feat32,
    const int32_t *__restrict__ e32,
    const int32_t *__restrict__ r32,
    const int32_t *__restrict__ fie32,
    const int32_t *__restrict__ lid32,
    const double *__restrict__ e_vals,
    const double *__restrict__ r_vals,
    const double *__restrict__ fie_vals,
    int n_iterations,
    int n_features,
    double inv_scaling,
    const double *__restrict__ table_s1,
    const double *__restrict__ table_s2,
    int table_stride,
    inter_weights::WeightCache &weight_cache,
    int max_order,
    int result_size,
    double *__restrict__ result)
{
    // Step 1: find leaf boundaries
    std::vector<int> leaf_start;
    leaf_start.reserve(n_iterations / 4);
    leaf_start.push_back(0);
    for (int i = 1; i < n_iterations; i++)
    {
        if (lid32[i] != lid32[i - 1])
            leaf_start.push_back(i);
    }
    leaf_start.push_back(n_iterations);
    int n_leaves = (int)leaf_start.size() - 1;

    // Step 2: parallel over leaves with thread-local result arrays
    #pragma omp parallel
    {
        double *local_result = new double[result_size]();
        #pragma omp for schedule(dynamic, 16)
        for (int leaf = 0; leaf < n_leaves; leaf++)
        {
            int start = leaf_start[leaf];
            int end = leaf_start[leaf + 1];

            for (int i = start; i < end; i++)
            {
                // Order-1 contribution
                double w1;
                if constexpr (IT == IndexType::BII)
                {
                    double sign = 1.0 - 2.0 * (1.0 - fie_vals[i]);
                    w1 = sign * exp2(-(e_vals[i] + r_vals[i] - 1.0));
                }
                else if constexpr (IT == IndexType::CUSTOM)
                {
                    w1 = (double)weight_cache.get_weight(
                        n_features, e32[i], r32[i], fie32[i], 1 - fie32[i], 1, IT, max_order);
                }
                else
                {
                    w1 = table_s1[fie32[i] * table_stride * table_stride + e32[i] * table_stride + r32[i]];
                }
                local_result[feat32[i]] += (double)(leaf_pred[i] * w1 * inv_scaling);

                // Order-2: pairwise interactions within the same leaf
                for (int j = i + 1; j < end; j++)
                {
                    int s_cap_e_c = fie32[i] + fie32[j];
                    double w2;
                    if constexpr (IT == IndexType::BII)
                    {
                        int s_cap_r_c = 2 - s_cap_e_c;
                        double sign_c = (s_cap_r_c % 2 == 0) ? 1.0 : -1.0;
                        w2 = sign_c * exp2(-(e_vals[i] + r_vals[i] - 2.0));
                    }
                    else if constexpr (IT == IndexType::CUSTOM)
                    {
                        int s_cap_r_c = 2 - s_cap_e_c;
                        w2 = (double)weight_cache.get_weight(
                            n_features, e32[i], r32[i], s_cap_e_c, s_cap_r_c, 2, IT, max_order);
                    }
                    else
                    {
                        w2 = table_s2[s_cap_e_c * table_stride * table_stride + e32[i] * table_stride + r32[i]];
                    }
                    // Inline interaction index computation (avoids O(n²) precomputed table)
                    int fi = feat32[i], fj = feat32[j];
                    if (fi > fj) std::swap(fi, fj);
                    int idx = (fi == fj) ? fi : n_features + (fi * n_features - fi * (fi + 1) / 2) + (fj - fi - 1);
                    local_result[idx] += (double)(leaf_pred[i] * w2 * inv_scaling);
                }
            }
        }
        // Merge thread-local results
        #pragma omp critical
        {
            for (int k = 0; k < result_size; k++)
                result[k] += local_result[k];
        }
        delete[] local_result;
    }
}

// Template: order-3 computation parallelized over leaves
template <IndexType IT>
static void compute_order3_leafparallel(
    const double *__restrict__ leaf_pred,
    const int32_t *__restrict__ feat32,
    const int32_t *__restrict__ e32,
    const int32_t *__restrict__ r32,
    const int32_t *__restrict__ fie32,
    const int32_t *__restrict__ lid32,
    const double *__restrict__ e_vals,
    const double *__restrict__ r_vals,
    const double *__restrict__ fie_vals,
    int n_iterations,
    int n_features,
    double inv_scaling,
    const double *__restrict__ table_s1,
    const double *__restrict__ table_s2,
    const double *__restrict__ table_s3,
    int table_stride,
    inter_weights::WeightCache &weight_cache,
    int max_order,
    int result_size,
    double *__restrict__ result)
{
    // Step 1: find leaf boundaries
    std::vector<int> leaf_start;
    leaf_start.reserve(n_iterations / 4);
    leaf_start.push_back(0);
    for (int i = 1; i < n_iterations; i++)
    {
        if (lid32[i] != lid32[i - 1])
            leaf_start.push_back(i);
    }
    leaf_start.push_back(n_iterations);
    int n_leaves = (int)leaf_start.size() - 1;

    // Step 2: parallel over leaves with thread-local result arrays
    #pragma omp parallel
    {
        double *local_result = new double[result_size]();
        #pragma omp for schedule(dynamic, 16)
        for (int leaf = 0; leaf < n_leaves; leaf++)
        {
            int start = leaf_start[leaf];
            int end = leaf_start[leaf + 1];

            for (int i = start; i < end; i++)
            {
                int fi = feat32[i];
                int ei = e32[i], ri = r32[i], fiei = fie32[i];
                double lp = leaf_pred[i] * inv_scaling;

                // Order-1 contribution
                double w1;
                if constexpr (IT == IndexType::BII)
                {
                    double sign = 1.0 - 2.0 * (1.0 - fie_vals[i]);
                    w1 = sign * exp2(-(e_vals[i] + r_vals[i] - 1.0));
                }
                else if constexpr (IT == IndexType::CUSTOM)
                {
                    w1 = (double)weight_cache.get_weight(
                        n_features, ei, ri, fiei, 1 - fiei, 1, IT, max_order);
                }
                else
                {
                    w1 = table_s1[fiei * table_stride * table_stride + ei * table_stride + ri];
                }
                local_result[fi] += (double)(lp * w1);

                // Order-2 and order-3: interactions within the same leaf
                for (int j = i + 1; j < end; j++)
                {
                    int fj = feat32[j];
                    int s_cap_e_2 = fiei + fie32[j];
                    double w2;
                    if constexpr (IT == IndexType::BII)
                    {
                        int s_cap_r_c = 2 - s_cap_e_2;
                        double sign_c = (s_cap_r_c % 2 == 0) ? 1.0 : -1.0;
                        w2 = sign_c * exp2(-(e_vals[i] + r_vals[i] - 2.0));
                    }
                    else if constexpr (IT == IndexType::CUSTOM)
                    {
                        int s_cap_r_c = 2 - s_cap_e_2;
                        w2 = (double)weight_cache.get_weight(
                            n_features, ei, ri, s_cap_e_2, s_cap_r_c, 2, IT, max_order);
                    }
                    else
                    {
                        w2 = table_s2[s_cap_e_2 * table_stride * table_stride + ei * table_stride + ri];
                    }
                    // Compute order-2 index (compact upper-triangle without diagonal)
                    int fi2 = fi, fj2 = fj;
                    if (fi2 > fj2) { int t = fi2; fi2 = fj2; fj2 = t; }
                    int idx2 = (fi2 == fj2) ? fi2 : n_features + (fi2 * n_features - fi2 * (fi2 + 1) / 2) + (fj2 - fi2 - 1);
                    local_result[idx2] += (double)(lp * w2);

                    // Order-3
                    for (int k = j + 1; k < end; k++)
                    {
                        int fk = feat32[k];
                        int s_cap_e_3 = s_cap_e_2 + fie32[k];
                        double w3;
                        if constexpr (IT == IndexType::BII)
                        {
                            int s_cap_r_3 = 3 - s_cap_e_3;
                            double sign_3 = (s_cap_r_3 % 2 == 0) ? 1.0 : -1.0;
                            w3 = sign_3 * exp2(-(e_vals[i] + r_vals[i] - 3.0));
                        }
                        else if constexpr (IT == IndexType::CUSTOM)
                        {
                            int s_cap_r_3 = 3 - s_cap_e_3;
                            w3 = (double)weight_cache.get_weight(
                                n_features, ei, ri, s_cap_e_3, s_cap_r_3, 3, IT, max_order);
                        }
                        else
                        {
                            w3 = table_s3[s_cap_e_3 * table_stride * table_stride + ei * table_stride + ri];
                        }
                        int idx3 = index3(fi, fj, fk, n_features);
                        local_result[idx3] += (double)(lp * w3);
                    }
                }
            }
        }
        // Merge thread-local results
        #pragma omp critical
        {
            for (int k = 0; k < result_size; k++)
                result[k] += local_result[k];
        }
        delete[] local_result;
    }
}

// Dispatch macro to instantiate templates for all index types
#define DISPATCH_INDEX_TYPE(FUNC, index_type, ...) \
    do { \
        switch (index_type) { \
        case IndexType::SII:  FUNC<IndexType::SII>(__VA_ARGS__); break; \
        case IndexType::BII:  FUNC<IndexType::BII>(__VA_ARGS__); break; \
        case IndexType::CHII: FUNC<IndexType::CHII>(__VA_ARGS__); break; \
        case IndexType::FBII: FUNC<IndexType::FBII>(__VA_ARGS__); break; \
        case IndexType::FSII: FUNC<IndexType::FSII>(__VA_ARGS__); break; \
        case IndexType::STII: FUNC<IndexType::STII>(__VA_ARGS__); break; \
        case IndexType::MOEBIUS: FUNC<IndexType::MOEBIUS>(__VA_ARGS__); break; \
        case IndexType::CUSTOM: FUNC<IndexType::CUSTOM>(__VA_ARGS__); break; \
        } \
    } while(0)

// === End optimized helpers ===

static PyObject *compute_interactions_flatten(PyObject *self, PyObject *args)
{
    /**
     * This function computes interactions for a single tree using a flattened representation of the tree structure, which is more memory efficient and can be faster to process.
     * The input parameters are similar to compute_interactions_batched_sparse, but instead of lists of numpy arrays for multiple trees, we have single numpy arrays that represent the flattened tree structure.
     * The function also supports an optional custom weight table for computing interactions.
     * The function should be used when the max_order is below or equal to 2.
     * The funcion has the following input parameters( in exact order):
     * - leaf_predictions: A numpy array containing the predictions at the leaf nodes of a decision tree.
     * - features: A numpy array containing the feature indices used for splitting at each decision node in a flattened tree representation.
     * - e_sizes: A numpy array containing the sizes of the subsets of features taken according to the point explained ("e") for each node in the flattened tree representation.
     * - r_sizes: A numpy array containing the sizes of the subsets of features taken according to the reference point ("r") set for each node in the flattened tree representation.
     * - features_in_e: A numy array indicating whether the currently observed feature is part of "e" or "r".
     * - leaf_id: A numpy array indicating to which leaf the corresponding feature belongs to. Necessary for order 2 computation.
     */

    PyObject *leaf_predictions_obj;
    PyObject *features_obj;
    PyObject *e_sizes;
    PyObject *r_sizes;
    PyObject *feature_in_e_obj;
    PyObject *leaf_id;
    const char *index_cptr;
    int n_iterations;
    int max_order;
    int verbose;
    int n_features;
    int e_length;
    double scaling_factor = 1.0;
    // Optional custom weight table (None when not used)
    PyObject *weight_table_obj = Py_None;
    IndexType index_type;
    if (!PyArg_ParseTuple(args, "OOOOOOsiiiiid|O", &leaf_predictions_obj, &features_obj, &e_sizes, &r_sizes, &feature_in_e_obj, &leaf_id, &index_cptr, &n_iterations, &n_features, &e_length, &max_order, &verbose, &scaling_factor, &weight_table_obj))
    {
        return NULL;
    }
    if (!PyArray_Check(leaf_predictions_obj) || !PyArray_Check(features_obj) || !PyArray_Check(e_sizes) || !PyArray_Check(r_sizes) || !PyArray_Check(feature_in_e_obj) || !PyArray_Check(leaf_id))
    {
        PyErr_SetString(PyExc_TypeError, "Input data must be numpy arrays");
        return NULL;
    }
    PyArrayObject *leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_predictions_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject *)PyArray_FROM_OTF(features_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *e_sizes_array = (PyArrayObject *)PyArray_FROM_OTF(e_sizes, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *r_sizes_array = (PyArrayObject *)PyArray_FROM_OTF(r_sizes, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *feature_in_e_array = (PyArrayObject *)PyArray_FROM_OTF(feature_in_e_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *leaf_id_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_id, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!leaf_predictions_array || !features_array || !e_sizes_array || !r_sizes_array || !feature_in_e_array || !leaf_id_array)
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(features_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        Py_XDECREF(feature_in_e_array);
        Py_XDECREF(leaf_id_array);
        PyErr_SetString(PyExc_TypeError, "Failed to convert input to numpy arrays");
        return NULL;
    }
    double *leaf_predictions = (double *)PyArray_DATA(leaf_predictions_array);
    int64_t *features = (int64_t *)PyArray_DATA(features_array);
    int64_t *e_sizes_data = (int64_t *)PyArray_DATA(e_sizes_array);
    int64_t *r_sizes_data = (int64_t *)PyArray_DATA(r_sizes_array);
    int64_t *feature_in_e_data = (int64_t *)PyArray_DATA(feature_in_e_array);
    int64_t *leaf_id_data = (int64_t *)PyArray_DATA(leaf_id_array);
    if (!parse_index_type(std::string(index_cptr), index_type))
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(features_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        Py_XDECREF(feature_in_e_array);
        Py_XDECREF(leaf_id_array);
        PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + std::string(index_cptr)).c_str());
        return NULL;
    }

    // Extract custom weight table pointer if provided
    const double *custom_table = nullptr;
    int64_t custom_N = 0, custom_K = 0;
    PyArrayObject *weight_table_array = nullptr;
    if (weight_table_obj != Py_None)
    {
        weight_table_array = (PyArrayObject *)PyArray_FROM_OTF(weight_table_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (!weight_table_array)
        {
            Py_XDECREF(leaf_predictions_array);
            Py_XDECREF(features_array);
            Py_XDECREF(e_sizes_array);
            Py_XDECREF(r_sizes_array);
            Py_XDECREF(feature_in_e_array);
            Py_XDECREF(leaf_id_array);
            PyErr_SetString(PyExc_TypeError, "weight_table must be a float64 numpy array");
            return NULL;
        }
        custom_table = (const double *)PyArray_DATA(weight_table_array);
        custom_N = (int64_t)n_features + 1;
        custom_K = (int64_t)max_order + 1;
    }
    // Weight cache (only needed for CUSTOM index type in the hot path)
    inter_weights::WeightCache weight_cache = (custom_table != nullptr)
                                                  ? inter_weights::WeightCache((uint64_t)(3 * n_features), custom_table, custom_N, custom_K)
                                                  : inter_weights::WeightCache((uint64_t)(3 * n_features));

    if (max_order <= 3)
    {
        // --- Phase 0: Convert int64 inputs to int32 + float64 for SIMD-friendly processing ---
        int32_t *feat32 = new int32_t[n_iterations];
        int32_t *e32 = new int32_t[n_iterations];
        int32_t *r32 = new int32_t[n_iterations];
        int32_t *fie32 = new int32_t[n_iterations];
        int32_t *lid32 = new int32_t[n_iterations];
        double *e_vals = new double[n_iterations];
        double *r_vals = new double[n_iterations];
        double *fie_vals = new double[n_iterations];
        for (int i = 0; i < n_iterations; i++)
        {
            feat32[i] = (int32_t)features[i];
            e32[i] = (int32_t)e_sizes_data[i];
            r32[i] = (int32_t)r_sizes_data[i];
            fie32[i] = (int32_t)feature_in_e_data[i];
            lid32[i] = (int32_t)leaf_id_data[i];
            e_vals[i] = (double)e_sizes_data[i];
            r_vals[i] = (double)r_sizes_data[i];
            fie_vals[i] = (double)feature_in_e_data[i];
        }

        double inv_scaling = 1.0 / scaling_factor;

        // --- Phase 1: Precompute weight lookup tables ---
        // Scan actual e/r bounds — table only needs to cover values that appear in data,
        // not all of [0, n_features]. For boolean trees with depth ~7, max e/r ≈ 7.
        int max_e = 0, max_r = 0;
        for (int i = 0; i < n_iterations; i++)
        {
            if (e32[i] > max_e) max_e = e32[i];
            if (r32[i] > max_r) max_r = r32[i];
        }
        int table_stride = std::max(max_e, max_r) + 1;

        double *table_s1 = nullptr;
        double *table_s2 = nullptr;
        double *table_s3 = nullptr;
        if (index_type != IndexType::CUSTOM)
        {
            table_s1 = new double[2 * table_stride * table_stride];
            if (max_order >= 2)
                table_s2 = new double[3 * table_stride * table_stride];
            if (max_order >= 3)
                table_s3 = new double[4 * table_stride * table_stride];
            precompute_weight_tables(index_type, n_features, max_order, table_s1, table_s2, table_s3, table_stride);
        }

        // --- Phase 2: Compute result ---
        int result_size = 0;
        for (int order = 1; order <= max_order; order++)
        {
            result_size += static_cast<int>(inter_weights::binom(n_features, order));
        }
        double *result = new double[result_size]();

        Py_BEGIN_ALLOW_THREADS

        if (max_order == 1)
        {
            DISPATCH_INDEX_TYPE(compute_order1_twopass, index_type,
                leaf_predictions, feat32, e32, r32, fie32,
                e_vals, r_vals, fie_vals,
                n_iterations, n_features, inv_scaling,
                table_s1, table_stride, weight_cache, max_order, result);
        }
        else if (max_order == 2)
        {
            DISPATCH_INDEX_TYPE(compute_order2_leafparallel, index_type,
                leaf_predictions, feat32, e32, r32, fie32, lid32,
                e_vals, r_vals, fie_vals,
                n_iterations, n_features, inv_scaling,
                table_s1, table_s2, table_stride,
                weight_cache, max_order, result_size, result);
        }
        else if (max_order == 3)
        {
            DISPATCH_INDEX_TYPE(compute_order3_leafparallel, index_type,
                leaf_predictions, feat32, e32, r32, fie32, lid32,
                e_vals, r_vals, fie_vals,
                n_iterations, n_features, inv_scaling,
                table_s1, table_s2, table_s3, table_stride,
                weight_cache, max_order, result_size, result);
        }

        Py_END_ALLOW_THREADS

        // --- Phase 3: Convert output to dict (sparse — skip zero entries) ---
        PyObject *output = PyDict_New();
        if (max_order == 1)
        {
            for (int i = 0; i < n_features; i++)
            {
                if (result[i] == 0.0) continue;
                PyObject *key = PyTuple_New(1);
                PyTuple_SetItem(key, 0, PyLong_FromLong(i));
                PyObject *value = PyFloat_FromDouble(result[i]);
                PyDict_SetItem(output, key, value);
                Py_DECREF(key);
                Py_DECREF(value);
            }
        }
        if (max_order >= 2)
        {
            // Main effects
            for (int i = 0; i < n_features; i++)
            {
                if (result[i] == 0.0) continue;
                PyObject *key = PyTuple_New(1);
                PyTuple_SetItem(key, 0, PyLong_FromLong(i));
                PyObject *value = PyFloat_FromDouble(result[i]);
                PyDict_SetItem(output, key, value);
                Py_DECREF(key);
                Py_DECREF(value);
            }
            // Pairwise interactions — forward iteration avoids while-loop reverse mapping
            {
                int pair_offset = 0;
                for (int pi = 0; pi < n_features; pi++)
                {
                    for (int pj = pi + 1; pj < n_features; pj++)
                    {
                        double v = result[n_features + pair_offset++];
                        if (v == 0.0) continue;
                        PyObject *key = PyTuple_New(2);
                        PyTuple_SET_ITEM(key, 0, PyLong_FromLong(pi));
                        PyTuple_SET_ITEM(key, 1, PyLong_FromLong(pj));
                        PyObject *value = PyFloat_FromDouble(v);
                        PyDict_SetItem(output, key, value);
                        Py_DECREF(key);
                        Py_DECREF(value);
                    }
                }
            }
        }
        if (max_order >= 3)
        {
            // Triple interactions — forward (kk,jj,ii) iteration avoids while-loop reverse mapping.
            // Compact layout: offset = ii + jj*(jj-1)/2 + kk*(kk-1)*(kk-2)/6, 0 <= ii < jj < kk.
            int base3 = n_features + n_features * (n_features - 1) / 2;
            int offset3 = 0;
            for (int kk = 2; kk < n_features; kk++)
            {
                for (int jj = 1; jj < kk; jj++)
                {
                    for (int ii = 0; ii < jj; ii++)
                    {
                        double v = result[base3 + offset3++];
                        if (v == 0.0) continue;
                        PyObject *key = PyTuple_New(3);
                        PyTuple_SET_ITEM(key, 0, PyLong_FromLong(ii));
                        PyTuple_SET_ITEM(key, 1, PyLong_FromLong(jj));
                        PyTuple_SET_ITEM(key, 2, PyLong_FromLong(kk));
                        PyObject *value = PyFloat_FromDouble(v);
                        PyDict_SetItem(output, key, value);
                        Py_DECREF(key);
                        Py_DECREF(value);
                    }
                }
            }
        }

        // --- Cleanup ---
        delete[] result;
        delete[] feat32;
        delete[] e32;
        delete[] r32;
        delete[] fie32;
        delete[] lid32;
        delete[] e_vals;
        delete[] r_vals;
        delete[] fie_vals;
        delete[] table_s1;
        delete[] table_s2;
        delete[] table_s3;

        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(features_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        Py_XDECREF(feature_in_e_array);
        Py_XDECREF(leaf_id_array);
        Py_XDECREF(weight_table_array);

        return output;
    }
    // max_order > 3 not supported by this function
    PyErr_SetString(PyExc_ValueError, "compute_interactions_flatten only supports max_order <= 3");
    Py_XDECREF(leaf_predictions_array);
    Py_XDECREF(features_array);
    Py_XDECREF(e_sizes_array);
    Py_XDECREF(r_sizes_array);
    Py_XDECREF(feature_in_e_array);
    Py_XDECREF(leaf_id_array);
    Py_XDECREF(weight_table_array);
    return NULL;
}

// === preprocess_boolean_trees ===
// DFS traversal of boolean trees using C++ BitSets.
// Produces the 6 flat numpy arrays needed by compute_interactions_flatten.
static PyObject *preprocess_boolean_trees(PyObject *self, PyObject *args)
{
    PyObject *values_list_obj;
    PyObject *features_list_obj;
    PyObject *children_left_list_obj;
    PyObject *children_right_list_obj;
    int n_features;

    if (!PyArg_ParseTuple(args, "OOOOi",
                          &values_list_obj, &features_list_obj,
                          &children_left_list_obj, &children_right_list_obj,
                          &n_features))
    {
        return NULL;
    }

    if (!PyList_Check(values_list_obj) || !PyList_Check(features_list_obj) ||
        !PyList_Check(children_left_list_obj) || !PyList_Check(children_right_list_obj))
    {
        PyErr_SetString(PyExc_TypeError, "All tree inputs must be lists of numpy arrays");
        return NULL;
    }

    Py_ssize_t num_trees = PyList_Size(values_list_obj);
    if (num_trees != PyList_Size(features_list_obj) ||
        num_trees != PyList_Size(children_left_list_obj) ||
        num_trees != PyList_Size(children_right_list_obj))
    {
        PyErr_SetString(PyExc_ValueError, "All tree lists must have the same length");
        return NULL;
    }

    // Output buffers (grow dynamically during DFS)
    std::vector<int64_t> features_out;
    std::vector<double> leaf_vals_out;
    std::vector<int64_t> e_sizes_out;
    std::vector<int64_t> r_sizes_out;
    std::vector<int64_t> fie_out;
    std::vector<int64_t> lid_out;

    // Reserve estimated space: ~64 leaves/tree × avg 6 features/leaf × num_trees
    size_t est = static_cast<size_t>(num_trees) * 64 * 6;
    features_out.reserve(est);
    leaf_vals_out.reserve(est);
    e_sizes_out.reserve(est);
    r_sizes_out.reserve(est);
    fie_out.reserve(est);
    lid_out.reserve(est);

    int64_t leaf_counter = 0;

    // Store converted arrays for cleanup
    std::vector<std::tuple<PyArrayObject *, PyArrayObject *, PyArrayObject *, PyArrayObject *>> arrays_for_decref;

    for (Py_ssize_t t = 0; t < num_trees; t++)
    {
        PyArrayObject *vals_arr = (PyArrayObject *)PyArray_FROM_OTF(
            PyList_GetItem(values_list_obj, t), NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *feat_arr = (PyArrayObject *)PyArray_FROM_OTF(
            PyList_GetItem(features_list_obj, t), NPY_INT64, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *cl_arr = (PyArrayObject *)PyArray_FROM_OTF(
            PyList_GetItem(children_left_list_obj, t), NPY_INT64, NPY_ARRAY_IN_ARRAY);
        PyArrayObject *cr_arr = (PyArrayObject *)PyArray_FROM_OTF(
            PyList_GetItem(children_right_list_obj, t), NPY_INT64, NPY_ARRAY_IN_ARRAY);

        if (!vals_arr || !feat_arr || !cl_arr || !cr_arr)
        {
            Py_XDECREF(vals_arr);
            Py_XDECREF(feat_arr);
            Py_XDECREF(cl_arr);
            Py_XDECREF(cr_arr);
            for (auto &arr_tuple : arrays_for_decref)
            {
                Py_XDECREF(std::get<0>(arr_tuple));
                Py_XDECREF(std::get<1>(arr_tuple));
                Py_XDECREF(std::get<2>(arr_tuple));
                Py_XDECREF(std::get<3>(arr_tuple));
            }
            PyErr_SetString(PyExc_TypeError, "Failed to convert tree arrays");
            return NULL;
        }
        arrays_for_decref.push_back(std::make_tuple(vals_arr, feat_arr, cl_arr, cr_arr));

        double *values = (double *)PyArray_DATA(vals_arr);
        int64_t *features = (int64_t *)PyArray_DATA(feat_arr);
        int64_t *children_left = (int64_t *)PyArray_DATA(cl_arr);
        int64_t *children_right = (int64_t *)PyArray_DATA(cr_arr);

        // DFS with BitSets
        // Stack entries: (node_id, E, R)
        std::vector<StackFrame> stack;
        stack.reserve(256);
        stack.push_back(StackFrame(0, BitSet(n_features), BitSet(n_features), 0, 0));

        while (!stack.empty())
        {
            StackFrame frame = std::move(stack.back());
            stack.pop_back();
            int64_t node_id = frame.node_id;

            bool is_leaf = (children_left[node_id] == children_right[node_id]);
            if (is_leaf)
            {
                double leaf_val = values[node_id];
                int64_t e_size = static_cast<int64_t>(frame.E.num_bits());
                int64_t r_size = static_cast<int64_t>(frame.R.num_bits());

                // Append E features (feature_in_E = 1)
                frame.E.for_each_set_bit([&](uint64_t feat)
                {
                    features_out.push_back(static_cast<int64_t>(feat));
                    leaf_vals_out.push_back(leaf_val);
                    e_sizes_out.push_back(e_size);
                    r_sizes_out.push_back(r_size);
                    fie_out.push_back(1);
                    lid_out.push_back(leaf_counter);
                });
                // Append R features (feature_in_E = 0)
                frame.R.for_each_set_bit([&](uint64_t feat)
                {
                    features_out.push_back(static_cast<int64_t>(feat));
                    leaf_vals_out.push_back(leaf_val);
                    e_sizes_out.push_back(e_size);
                    r_sizes_out.push_back(r_size);
                    fie_out.push_back(0);
                    lid_out.push_back(leaf_counter);
                });
                leaf_counter++;
                continue;
            }

            int64_t feature = features[node_id];

            // Go left: feature → R (unless already in E)
            if (!frame.E.contains(feature))
            {
                BitSet next_R = frame.R;
                next_R.add(feature);
                stack.push_back(StackFrame(
                    children_left[node_id], frame.E, next_R,
                    frame.e, frame.r + 1));
            }
            // Go right: feature → E (unless already in R)
            if (!frame.R.contains(feature))
            {
                BitSet next_E = frame.E;
                next_E.add(feature);
                stack.push_back(StackFrame(
                    children_right[node_id], next_E, frame.R,
                    frame.e + 1, frame.r));
            }
        }
    }

    // Cleanup tree arrays
    for (auto &arr_tuple : arrays_for_decref)
    {
        Py_XDECREF(std::get<0>(arr_tuple));
        Py_XDECREF(std::get<1>(arr_tuple));
        Py_XDECREF(std::get<2>(arr_tuple));
        Py_XDECREF(std::get<3>(arr_tuple));
    }

    // Convert output vectors to numpy arrays
    npy_intp n_total = static_cast<npy_intp>(features_out.size());

    PyObject *np_features = PyArray_SimpleNew(1, &n_total, NPY_INT64);
    PyObject *np_leaf_vals = PyArray_SimpleNew(1, &n_total, NPY_FLOAT64);
    PyObject *np_e_sizes = PyArray_SimpleNew(1, &n_total, NPY_INT64);
    PyObject *np_r_sizes = PyArray_SimpleNew(1, &n_total, NPY_INT64);
    PyObject *np_fie = PyArray_SimpleNew(1, &n_total, NPY_INT64);
    PyObject *np_lid = PyArray_SimpleNew(1, &n_total, NPY_INT64);

    if (!np_features || !np_leaf_vals || !np_e_sizes || !np_r_sizes || !np_fie || !np_lid)
    {
        Py_XDECREF(np_features);
        Py_XDECREF(np_leaf_vals);
        Py_XDECREF(np_e_sizes);
        Py_XDECREF(np_r_sizes);
        Py_XDECREF(np_fie);
        Py_XDECREF(np_lid);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    if (n_total > 0)
    {
        memcpy(PyArray_DATA((PyArrayObject *)np_features), features_out.data(), n_total * sizeof(int64_t));
        memcpy(PyArray_DATA((PyArrayObject *)np_leaf_vals), leaf_vals_out.data(), n_total * sizeof(double));
        memcpy(PyArray_DATA((PyArrayObject *)np_e_sizes), e_sizes_out.data(), n_total * sizeof(int64_t));
        memcpy(PyArray_DATA((PyArrayObject *)np_r_sizes), r_sizes_out.data(), n_total * sizeof(int64_t));
        memcpy(PyArray_DATA((PyArrayObject *)np_fie), fie_out.data(), n_total * sizeof(int64_t));
        memcpy(PyArray_DATA((PyArrayObject *)np_lid), lid_out.data(), n_total * sizeof(int64_t));
    }

    // Return tuple of 6 arrays
    PyObject *result = PyTuple_New(6);
    PyTuple_SetItem(result, 0, np_features);     // E_R_flatten
    PyTuple_SetItem(result, 1, np_leaf_vals);     // leaf_vals_flatten
    PyTuple_SetItem(result, 2, np_e_sizes);       // e_size_flatten
    PyTuple_SetItem(result, 3, np_r_sizes);       // r_size_flatten
    PyTuple_SetItem(result, 4, np_fie);           // feature_in_E
    PyTuple_SetItem(result, 5, np_lid);           // leaf_id

    return result;
}

// === preprocess_trees_point ===
// C port of InterventionalTreeSHAPIQ._preprocess_tree / obtain_E_R_values_point:
// for every reference sample and every tree, DFS the tree routing the explain
// point vs. the reference point, and emit the 6 flat arrays consumed by
// compute_interactions_flatten. Row layout per leaf: E features ascending, then
// R features ascending (BitSet iteration order matches the sorted numpy sets).
static PyObject *preprocess_trees_point(PyObject *self, PyObject *args)
{
    PyObject *values_list_obj;
    PyObject *thresholds_list_obj;
    PyObject *features_list_obj;
    PyObject *children_left_list_obj;
    PyObject *children_right_list_obj;
    PyObject *children_missing_list_obj;
    PyObject *reference_data_obj;
    PyObject *explain_point_obj;
    const char *decision_type_cptr;
    // Optional categorical split CSR lists (None for numeric-only ensembles)
    PyObject *cat_values_obj = Py_None;
    PyObject *cat_start_obj = Py_None;
    PyObject *cat_size_obj = Py_None;

    if (!PyArg_ParseTuple(args, "OOOOOOOOs|OOO",
                          &values_list_obj, &thresholds_list_obj, &features_list_obj,
                          &children_left_list_obj, &children_right_list_obj,
                          &children_missing_list_obj,
                          &reference_data_obj, &explain_point_obj,
                          &decision_type_cptr,
                          &cat_values_obj, &cat_start_obj, &cat_size_obj))
    {
        return NULL;
    }

    const bool use_categorical = (cat_values_obj != Py_None);
    if (!PyList_Check(values_list_obj) || !PyList_Check(thresholds_list_obj) ||
        !PyList_Check(features_list_obj) || !PyList_Check(children_left_list_obj) ||
        !PyList_Check(children_right_list_obj) || !PyList_Check(children_missing_list_obj))
    {
        PyErr_SetString(PyExc_TypeError, "All tree inputs must be lists of numpy arrays");
        return NULL;
    }
    if (use_categorical && (!PyList_Check(cat_values_obj) || !PyList_Check(cat_start_obj) || !PyList_Check(cat_size_obj)))
    {
        PyErr_SetString(PyExc_TypeError, "cat_values, cat_start, and cat_size must be lists of numpy arrays (or None)");
        return NULL;
    }

    Py_ssize_t num_trees = PyList_Size(values_list_obj);
    if (num_trees != PyList_Size(thresholds_list_obj) ||
        num_trees != PyList_Size(features_list_obj) ||
        num_trees != PyList_Size(children_left_list_obj) ||
        num_trees != PyList_Size(children_right_list_obj) ||
        num_trees != PyList_Size(children_missing_list_obj) ||
        (use_categorical && (num_trees != PyList_Size(cat_values_obj) ||
                             num_trees != PyList_Size(cat_start_obj) ||
                             num_trees != PyList_Size(cat_size_obj))))
    {
        PyErr_SetString(PyExc_ValueError, "All tree lists must have the same length");
        return NULL;
    }

    // Every converted array is tracked here; the Tree structs hold raw pointers
    // into them, so they are released only after the DFS is done.
    std::vector<PyArrayObject *> arrays_for_decref;
    auto cleanup_arrays = [&arrays_for_decref]()
    {
        for (PyArrayObject *arr : arrays_for_decref)
            Py_XDECREF(arr);
    };
    auto convert_item = [&arrays_for_decref](PyObject *list_obj, Py_ssize_t idx, int np_type) -> PyArrayObject *
    {
        PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
            PyList_GetItem(list_obj, idx), np_type, NPY_ARRAY_IN_ARRAY);
        if (arr)
            arrays_for_decref.push_back(arr);
        return arr;
    };

    PyArrayObject *reference_data_array = (PyArrayObject *)PyArray_FROM_OTF(reference_data_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (reference_data_array)
        arrays_for_decref.push_back(reference_data_array);
    PyArrayObject *explain_point_array = (PyArrayObject *)PyArray_FROM_OTF(explain_point_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (explain_point_array)
        arrays_for_decref.push_back(explain_point_array);
    if (!reference_data_array || !explain_point_array || PyArray_NDIM(reference_data_array) != 2)
    {
        cleanup_arrays();
        PyErr_SetString(PyExc_TypeError, "reference_data must be a 2-D float64 numpy array and explain_point a float64 numpy array");
        return NULL;
    }

    const double *reference_data = (const double *)PyArray_DATA(reference_data_array);
    const double *explain_data = (const double *)PyArray_DATA(explain_point_array);
    const int64_t n_ref = (int64_t)PyArray_DIM(reference_data_array, 0);
    const int64_t n_features = (int64_t)PyArray_DIM(reference_data_array, 1);

    std::vector<Tree> trees;
    trees.reserve(static_cast<size_t>(num_trees));
    for (Py_ssize_t t = 0; t < num_trees; t++)
    {
        PyArrayObject *vals_arr = convert_item(values_list_obj, t, NPY_FLOAT64);
        PyArrayObject *thr_arr = convert_item(thresholds_list_obj, t, NPY_FLOAT64);
        PyArrayObject *feat_arr = convert_item(features_list_obj, t, NPY_INT64);
        PyArrayObject *cl_arr = convert_item(children_left_list_obj, t, NPY_INT64);
        PyArrayObject *cr_arr = convert_item(children_right_list_obj, t, NPY_INT64);
        PyArrayObject *cm_arr = convert_item(children_missing_list_obj, t, NPY_BOOL);
        PyArrayObject *cat_values_arr = NULL, *cat_start_arr = NULL, *cat_size_arr = NULL;
        if (use_categorical)
        {
            cat_values_arr = convert_item(cat_values_obj, t, NPY_INT64);
            cat_start_arr = convert_item(cat_start_obj, t, NPY_INT64);
            cat_size_arr = convert_item(cat_size_obj, t, NPY_INT64);
        }
        if (!vals_arr || !thr_arr || !feat_arr || !cl_arr || !cr_arr || !cm_arr ||
            (use_categorical && (!cat_values_arr || !cat_start_arr || !cat_size_arr)))
        {
            cleanup_arrays();
            PyErr_SetString(PyExc_TypeError, "Failed to convert tree arrays");
            return NULL;
        }
        trees.push_back(Tree(
            (double *)PyArray_DATA(vals_arr),
            (double *)PyArray_DATA(thr_arr),
            (int64_t *)PyArray_DATA(feat_arr),
            (int64_t *)PyArray_DATA(cl_arr),
            (int64_t *)PyArray_DATA(cr_arr),
            (bool *)PyArray_DATA(cm_arr),
            std::string(decision_type_cptr),
            use_categorical ? (const int64_t *)PyArray_DATA(cat_values_arr) : nullptr,
            use_categorical ? (const int64_t *)PyArray_DATA(cat_start_arr) : nullptr,
            use_categorical ? (const int64_t *)PyArray_DATA(cat_size_arr) : nullptr));
    }

    // Output buffers (one row per (leaf, feature in E or R) pair)
    std::vector<int64_t> features_out;
    std::vector<double> leaf_vals_out;
    std::vector<int64_t> e_sizes_out;
    std::vector<int64_t> r_sizes_out;
    std::vector<int64_t> fie_out;
    std::vector<int64_t> lid_out;

    // Reserve estimated space: ~32 (leaf, feature) rows per (reference, tree) pair
    size_t est = static_cast<size_t>(n_ref) * static_cast<size_t>(num_trees) * 32;
    features_out.reserve(est);
    leaf_vals_out.reserve(est);
    e_sizes_out.reserve(est);
    r_sizes_out.reserve(est);
    fie_out.reserve(est);
    lid_out.reserve(est);

    int64_t leaf_counter = 0;

    Py_BEGIN_ALLOW_THREADS
    std::vector<StackFrame> stack;
    stack.reserve(256);
    BitSet empty_set(n_features);

    // Match the Python loop order: reference samples outer, trees inner.
    for (int64_t i = 0; i < n_ref; i++)
    {
        const double *reference_sample = reference_data + i * n_features;
        for (Py_ssize_t t = 0; t < num_trees; t++)
        {
            Tree &tree = trees[t];
            stack.push_back(StackFrame(0, empty_set, empty_set, 0, 0));
            while (!stack.empty())
            {
                StackFrame frame = std::move(stack.back());
                stack.pop_back();
                int64_t node_id = frame.node_id;

                bool is_leaf = (tree.children_left[node_id] == tree.children_right[node_id]);
                if (is_leaf)
                {
                    const double leaf_val = tree.leaf_predictions[node_id];
                    const int64_t e_size = static_cast<int64_t>(frame.E.num_bits());
                    const int64_t r_size = static_cast<int64_t>(frame.R.num_bits());

                    // Append E features (feature_in_E = 1)
                    frame.E.for_each_set_bit([&](uint64_t feat)
                    {
                        features_out.push_back(static_cast<int64_t>(feat));
                        leaf_vals_out.push_back(leaf_val);
                        e_sizes_out.push_back(e_size);
                        r_sizes_out.push_back(r_size);
                        fie_out.push_back(1);
                        lid_out.push_back(leaf_counter);
                    });
                    // Append R features (feature_in_E = 0)
                    frame.R.for_each_set_bit([&](uint64_t feat)
                    {
                        features_out.push_back(static_cast<int64_t>(feat));
                        leaf_vals_out.push_back(leaf_val);
                        e_sizes_out.push_back(e_size);
                        r_sizes_out.push_back(r_size);
                        fie_out.push_back(0);
                        lid_out.push_back(leaf_counter);
                    });
                    leaf_counter++;
                    continue;
                }

                const int64_t feature = tree.features[node_id];
                const int64_t child_explain = tree.goes_left(explain_data[feature], node_id)
                                                  ? tree.children_left[node_id]
                                                  : tree.children_right[node_id];
                const int64_t child_ref = tree.goes_left(reference_sample[feature], node_id)
                                              ? tree.children_left[node_id]
                                              : tree.children_right[node_id];

                if (child_explain != child_ref)
                {
                    if (!frame.R.contains(feature)) // feature is not fixed by the reference point
                    {
                        BitSet next_E = frame.E;
                        next_E.add(feature);
                        stack.push_back(StackFrame(child_explain, next_E, frame.R));
                    }
                    if (!frame.E.contains(feature)) // feature is not fixed by the explain point
                    {
                        BitSet next_R = frame.R;
                        next_R.add(feature);
                        stack.push_back(StackFrame(child_ref, frame.E, next_R));
                    }
                }
                else
                {
                    stack.push_back(StackFrame(child_explain, frame.E, frame.R));
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    cleanup_arrays();

    // Convert output vectors to numpy arrays
    npy_intp n_total = static_cast<npy_intp>(features_out.size());

    PyObject *np_features = PyArray_SimpleNew(1, &n_total, NPY_INT64);
    PyObject *np_leaf_vals = PyArray_SimpleNew(1, &n_total, NPY_FLOAT64);
    PyObject *np_e_sizes = PyArray_SimpleNew(1, &n_total, NPY_INT64);
    PyObject *np_r_sizes = PyArray_SimpleNew(1, &n_total, NPY_INT64);
    PyObject *np_fie = PyArray_SimpleNew(1, &n_total, NPY_INT64);
    PyObject *np_lid = PyArray_SimpleNew(1, &n_total, NPY_INT64);

    if (!np_features || !np_leaf_vals || !np_e_sizes || !np_r_sizes || !np_fie || !np_lid)
    {
        Py_XDECREF(np_features);
        Py_XDECREF(np_leaf_vals);
        Py_XDECREF(np_e_sizes);
        Py_XDECREF(np_r_sizes);
        Py_XDECREF(np_fie);
        Py_XDECREF(np_lid);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    if (n_total > 0)
    {
        memcpy(PyArray_DATA((PyArrayObject *)np_features), features_out.data(), n_total * sizeof(int64_t));
        memcpy(PyArray_DATA((PyArrayObject *)np_leaf_vals), leaf_vals_out.data(), n_total * sizeof(double));
        memcpy(PyArray_DATA((PyArrayObject *)np_e_sizes), e_sizes_out.data(), n_total * sizeof(int64_t));
        memcpy(PyArray_DATA((PyArrayObject *)np_r_sizes), r_sizes_out.data(), n_total * sizeof(int64_t));
        memcpy(PyArray_DATA((PyArrayObject *)np_fie), fie_out.data(), n_total * sizeof(int64_t));
        memcpy(PyArray_DATA((PyArrayObject *)np_lid), lid_out.data(), n_total * sizeof(int64_t));
    }

    // Return tuple of 6 arrays
    PyObject *result = PyTuple_New(6);
    PyTuple_SetItem(result, 0, np_features);      // E_R_flatten
    PyTuple_SetItem(result, 1, np_leaf_vals);     // leaf_vals_flatten
    PyTuple_SetItem(result, 2, np_e_sizes);       // e_size_flatten
    PyTuple_SetItem(result, 3, np_r_sizes);       // r_size_flatten
    PyTuple_SetItem(result, 4, np_fie);           // feature_in_E
    PyTuple_SetItem(result, 5, np_lid);           // leaf_id

    return result;
}

// === compute_interactions_cohort ===
// Fused dense kernel: one DFS per tree carries the COHORT of reference samples
// that reached the current (node, E, R) state, instead of one DFS per reference.
// At each split the cohort is partitioned into references that route like the
// explain point (E/R unchanged) and references that diverge (spawning the same
// E- and R-branches obtain_E_R_values_point creates per reference). At a leaf,
// all m cohort members contribute the identical (E, R, leaf_value) term, so the
// dense interaction buffer is updated ONCE with value scaled by m. Exact — the
// per-leaf weights depend only on (|E|, |R|, s_cap_e), never on the reference.

// Longest root-to-leaf path measured in internal nodes: bounds |E| + |R|, so the
// weight tables only need this stride instead of n_features + 1.
static int cohort_max_depth(const Tree &tree)
{
    std::vector<std::pair<int64_t, int>> stack;
    stack.push_back({0, 0});
    int best = 0;
    while (!stack.empty())
    {
        auto [node_id, depth] = stack.back();
        stack.pop_back();
        if (tree.children_left[node_id] == tree.children_right[node_id])
        {
            best = std::max(best, depth);
            continue;
        }
        stack.push_back({tree.children_left[node_id], depth + 1});
        stack.push_back({tree.children_right[node_id], depth + 1});
    }
    return best;
}

struct CohortFrame
{
    CohortFrame(int64_t node_id, const BitSet &E, const BitSet &R, int begin, int end)
        : node_id(node_id), E(E), R(R), begin(begin), end(end)
    {
    }
    int64_t node_id;
    BitSet E;
    BitSet R;
    int begin; // slice [begin, end) into the per-tree reference index buffer
    int end;
};

// Per-leaf dense update: identical math to compute_order{1,2,3}_leafparallel, but
// fed from the leaf's (feature, in_E) pairs directly instead of flat rows, and with
// the leaf value pre-scaled by the cohort size.
template <IndexType IT>
static inline void cohort_leaf_update(
    const int32_t *__restrict__ leaf_feats,
    const int32_t *__restrict__ leaf_fie,
    int n_leaf_feats,
    int e, int r,
    double scaled_val, // leaf_prediction * cohort_size * inv_scaling
    int n_features,
    const double *__restrict__ table_s1,
    const double *__restrict__ table_s2,
    const double *__restrict__ table_s3,
    int table_stride,
    inter_weights::WeightCache &weight_cache,
    int max_order,
    double *__restrict__ local_result)
{
    const double er = (double)(e + r);
    for (int i = 0; i < n_leaf_feats; i++)
    {
        const int fi = leaf_feats[i];
        const int fiei = leaf_fie[i];

        // Order-1 contribution
        double w1;
        if constexpr (IT == IndexType::BII)
        {
            double sign = 1.0 - 2.0 * (1.0 - (double)fiei);
            w1 = sign * exp2(-(er - 1.0));
        }
        else if constexpr (IT == IndexType::CUSTOM)
        {
            w1 = (double)weight_cache.get_weight(
                n_features, e, r, fiei, 1 - fiei, 1, IT, max_order);
        }
        else
        {
            w1 = table_s1[fiei * table_stride * table_stride + e * table_stride + r];
        }
        local_result[fi] += scaled_val * w1;

        if (max_order < 2)
            continue;

        // Order-2 (and order-3): interactions within the same leaf
        for (int j = i + 1; j < n_leaf_feats; j++)
        {
            const int fj = leaf_feats[j];
            const int s_cap_e_2 = fiei + leaf_fie[j];
            double w2;
            if constexpr (IT == IndexType::BII)
            {
                int s_cap_r_2 = 2 - s_cap_e_2;
                double sign_2 = (s_cap_r_2 % 2 == 0) ? 1.0 : -1.0;
                w2 = sign_2 * exp2(-(er - 2.0));
            }
            else if constexpr (IT == IndexType::CUSTOM)
            {
                int s_cap_r_2 = 2 - s_cap_e_2;
                w2 = (double)weight_cache.get_weight(
                    n_features, e, r, s_cap_e_2, s_cap_r_2, 2, IT, max_order);
            }
            else
            {
                w2 = table_s2[s_cap_e_2 * table_stride * table_stride + e * table_stride + r];
            }
            // Compact upper-triangle index; leaf features arrive unsorted (E then R)
            int fi2 = fi, fj2 = fj;
            if (fi2 > fj2) std::swap(fi2, fj2);
            int idx2 = n_features + (fi2 * n_features - fi2 * (fi2 + 1) / 2) + (fj2 - fi2 - 1);
            local_result[idx2] += scaled_val * w2;

            if (max_order < 3)
                continue;

            // Order-3
            for (int k = j + 1; k < n_leaf_feats; k++)
            {
                const int fk = leaf_feats[k];
                const int s_cap_e_3 = s_cap_e_2 + leaf_fie[k];
                double w3;
                if constexpr (IT == IndexType::BII)
                {
                    int s_cap_r_3 = 3 - s_cap_e_3;
                    double sign_3 = (s_cap_r_3 % 2 == 0) ? 1.0 : -1.0;
                    w3 = sign_3 * exp2(-(er - 3.0));
                }
                else if constexpr (IT == IndexType::CUSTOM)
                {
                    int s_cap_r_3 = 3 - s_cap_e_3;
                    w3 = (double)weight_cache.get_weight(
                        n_features, e, r, s_cap_e_3, s_cap_r_3, 3, IT, max_order);
                }
                else
                {
                    w3 = table_s3[s_cap_e_3 * table_stride * table_stride + e * table_stride + r];
                }
                local_result[index3(fi, fj, fk, n_features)] += scaled_val * w3;
            }
        }
    }
}

// Cohort DFS over one tree, accumulating into local_result.
// ref_order is a scratch permutation of [0, n_ref); slices are partitioned in
// place. This is safe because a split hands the SAME slice to both divergence
// branches: LIFO order fully explores one branch (which only permutes members
// within the slice) before the other starts, and set membership is preserved.
template <IndexType IT>
static void cohort_process_tree(
    Tree &tree,
    const double *__restrict__ reference_data,
    const double *__restrict__ explain_data,
    int n_ref,
    int n_features,
    double inv_scaling,
    const double *__restrict__ table_s1,
    const double *__restrict__ table_s2,
    const double *__restrict__ table_s3,
    int table_stride,
    inter_weights::WeightCache &weight_cache,
    int max_order,
    std::vector<int> &ref_order,       // scratch, size n_ref
    std::vector<int32_t> &leaf_feats,  // scratch, size >= max depth
    std::vector<int32_t> &leaf_fie,    // scratch, size >= max depth
    std::vector<CohortFrame> &stack,   // scratch
    double *__restrict__ local_result)
{
    for (int i = 0; i < n_ref; i++)
        ref_order[i] = i;

    BitSet empty_set(n_features);
    stack.clear();
    stack.push_back(CohortFrame(0, empty_set, empty_set, 0, n_ref));

    while (!stack.empty())
    {
        CohortFrame frame = std::move(stack.back());
        stack.pop_back();
        const int64_t node_id = frame.node_id;

        bool is_leaf = (tree.children_left[node_id] == tree.children_right[node_id]);
        if (is_leaf)
        {
            const int cohort_size = frame.end - frame.begin;
            const int e = (int)frame.E.num_bits();
            const int r = (int)frame.R.num_bits();
            if (e + r == 0)
                continue; // no constrained features -> no rows (baseline-only leaf)

            int n_leaf_feats = 0;
            frame.E.for_each_set_bit([&](uint64_t feat)
            {
                leaf_feats[n_leaf_feats] = (int32_t)feat;
                leaf_fie[n_leaf_feats] = 1;
                n_leaf_feats++;
            });
            frame.R.for_each_set_bit([&](uint64_t feat)
            {
                leaf_feats[n_leaf_feats] = (int32_t)feat;
                leaf_fie[n_leaf_feats] = 0;
                n_leaf_feats++;
            });

            const double scaled_val = tree.leaf_predictions[node_id] * (double)cohort_size * inv_scaling;
            cohort_leaf_update<IT>(
                leaf_feats.data(), leaf_fie.data(), n_leaf_feats, e, r, scaled_val,
                n_features, table_s1, table_s2, table_s3, table_stride,
                weight_cache, max_order, local_result);
            continue;
        }

        const int64_t feature = tree.features[node_id];
        const bool explain_left = tree.goes_left(explain_data[feature], node_id);
        const int64_t child_explain = explain_left ? tree.children_left[node_id] : tree.children_right[node_id];
        const int64_t child_other = explain_left ? tree.children_right[node_id] : tree.children_left[node_id];

        // Partition the cohort: [begin, mid) routes like the explain point, [mid, end) diverges.
        const double *ref_col = reference_data + feature;
        int *slice_begin = ref_order.data() + frame.begin;
        int *slice_end = ref_order.data() + frame.end;
        int *mid_ptr = std::partition(slice_begin, slice_end, [&](int ref_idx)
        {
            return tree.goes_left(ref_col[(int64_t)ref_idx * n_features], node_id) == explain_left;
        });
        const int mid = frame.begin + (int)(mid_ptr - slice_begin);

        if (mid > frame.begin) // agreeing cohort: descend with E, R unchanged
        {
            stack.push_back(CohortFrame(child_explain, frame.E, frame.R, frame.begin, mid));
        }
        if (mid < frame.end) // diverging cohort: same two branches as the per-reference DFS
        {
            if (!frame.R.contains(feature)) // feature is not fixed by the reference point
            {
                BitSet next_E = frame.E;
                next_E.add(feature);
                stack.push_back(CohortFrame(child_explain, next_E, frame.R, mid, frame.end));
            }
            if (!frame.E.contains(feature)) // feature is not fixed by the explain point
            {
                BitSet next_R = frame.R;
                next_R.add(feature);
                stack.push_back(CohortFrame(child_other, frame.E, next_R, mid, frame.end));
            }
        }
    }
}

// Dense result buffer -> {tuple: value} dict, skipping zeros.
// Same layout as compute_interactions_flatten's output conversion.
static PyObject *cohort_result_to_dict(const double *result, int n_features, int max_order)
{
    PyObject *output = PyDict_New();
    if (!output)
        return NULL;
    for (int i = 0; i < n_features; i++)
    {
        if (result[i] == 0.0) continue;
        PyObject *key = PyTuple_New(1);
        PyTuple_SET_ITEM(key, 0, PyLong_FromLong(i));
        PyObject *value = PyFloat_FromDouble(result[i]);
        PyDict_SetItem(output, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
    }
    if (max_order >= 2)
    {
        int pair_offset = 0;
        for (int pi = 0; pi < n_features; pi++)
        {
            for (int pj = pi + 1; pj < n_features; pj++)
            {
                double v = result[n_features + pair_offset++];
                if (v == 0.0) continue;
                PyObject *key = PyTuple_New(2);
                PyTuple_SET_ITEM(key, 0, PyLong_FromLong(pi));
                PyTuple_SET_ITEM(key, 1, PyLong_FromLong(pj));
                PyObject *value = PyFloat_FromDouble(v);
                PyDict_SetItem(output, key, value);
                Py_DECREF(key);
                Py_DECREF(value);
            }
        }
    }
    if (max_order >= 3)
    {
        int base3 = n_features + n_features * (n_features - 1) / 2;
        int offset3 = 0;
        for (int kk = 2; kk < n_features; kk++)
        {
            for (int jj = 1; jj < kk; jj++)
            {
                for (int ii = 0; ii < jj; ii++)
                {
                    double v = result[base3 + offset3++];
                    if (v == 0.0) continue;
                    PyObject *key = PyTuple_New(3);
                    PyTuple_SET_ITEM(key, 0, PyLong_FromLong(ii));
                    PyTuple_SET_ITEM(key, 1, PyLong_FromLong(jj));
                    PyTuple_SET_ITEM(key, 2, PyLong_FromLong(kk));
                    PyObject *value = PyFloat_FromDouble(v);
                    PyDict_SetItem(output, key, value);
                    Py_DECREF(key);
                    Py_DECREF(value);
                }
            }
        }
    }
    return output;
}

static PyObject *compute_interactions_cohort(PyObject *self, PyObject *args)
{
    PyObject *values_list_obj;
    PyObject *thresholds_list_obj;
    PyObject *features_list_obj;
    PyObject *children_left_list_obj;
    PyObject *children_right_list_obj;
    PyObject *children_missing_list_obj;
    PyObject *reference_data_obj;
    PyObject *explain_point_obj;
    const char *decision_type_cptr;
    const char *index_cptr;
    int max_order;
    int verbose;
    // Optional custom weight table and categorical split CSR lists (None when unused)
    PyObject *weight_table_obj = Py_None;
    PyObject *cat_values_obj = Py_None;
    PyObject *cat_start_obj = Py_None;
    PyObject *cat_size_obj = Py_None;

    if (!PyArg_ParseTuple(args, "OOOOOOOOssii|OOOO",
                          &values_list_obj, &thresholds_list_obj, &features_list_obj,
                          &children_left_list_obj, &children_right_list_obj,
                          &children_missing_list_obj,
                          &reference_data_obj, &explain_point_obj,
                          &decision_type_cptr, &index_cptr,
                          &max_order, &verbose,
                          &weight_table_obj, &cat_values_obj, &cat_start_obj, &cat_size_obj))
    {
        return NULL;
    }

    if (max_order < 1 || max_order > 3)
    {
        PyErr_SetString(PyExc_ValueError, "compute_interactions_cohort only supports 1 <= max_order <= 3");
        return NULL;
    }

    const bool use_categorical = (cat_values_obj != Py_None);
    if (!PyList_Check(values_list_obj) || !PyList_Check(thresholds_list_obj) ||
        !PyList_Check(features_list_obj) || !PyList_Check(children_left_list_obj) ||
        !PyList_Check(children_right_list_obj) || !PyList_Check(children_missing_list_obj))
    {
        PyErr_SetString(PyExc_TypeError, "All tree inputs must be lists of numpy arrays");
        return NULL;
    }
    if (use_categorical && (!PyList_Check(cat_values_obj) || !PyList_Check(cat_start_obj) || !PyList_Check(cat_size_obj)))
    {
        PyErr_SetString(PyExc_TypeError, "cat_values, cat_start, and cat_size must be lists of numpy arrays (or None)");
        return NULL;
    }

    Py_ssize_t num_trees = PyList_Size(values_list_obj);
    if (num_trees != PyList_Size(thresholds_list_obj) ||
        num_trees != PyList_Size(features_list_obj) ||
        num_trees != PyList_Size(children_left_list_obj) ||
        num_trees != PyList_Size(children_right_list_obj) ||
        num_trees != PyList_Size(children_missing_list_obj) ||
        (use_categorical && (num_trees != PyList_Size(cat_values_obj) ||
                             num_trees != PyList_Size(cat_start_obj) ||
                             num_trees != PyList_Size(cat_size_obj))))
    {
        PyErr_SetString(PyExc_ValueError, "All tree lists must have the same length");
        return NULL;
    }
    if (num_trees == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Input lists of numpy arrays must not be empty");
        return NULL;
    }

    IndexType index_type;
    if (!parse_index_type(std::string(index_cptr), index_type))
    {
        PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + std::string(index_cptr)).c_str());
        return NULL;
    }

    // Every converted array is tracked here; the Tree structs hold raw pointers
    // into them, so they are released only after the DFS is done.
    std::vector<PyArrayObject *> arrays_for_decref;
    auto cleanup_arrays = [&arrays_for_decref]()
    {
        for (PyArrayObject *arr : arrays_for_decref)
            Py_XDECREF(arr);
    };
    auto convert_item = [&arrays_for_decref](PyObject *list_obj, Py_ssize_t idx, int np_type) -> PyArrayObject *
    {
        PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
            PyList_GetItem(list_obj, idx), np_type, NPY_ARRAY_IN_ARRAY);
        if (arr)
            arrays_for_decref.push_back(arr);
        return arr;
    };

    PyArrayObject *reference_data_array = (PyArrayObject *)PyArray_FROM_OTF(reference_data_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (reference_data_array)
        arrays_for_decref.push_back(reference_data_array);
    PyArrayObject *explain_point_array = (PyArrayObject *)PyArray_FROM_OTF(explain_point_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (explain_point_array)
        arrays_for_decref.push_back(explain_point_array);
    if (!reference_data_array || !explain_point_array || PyArray_NDIM(reference_data_array) != 2)
    {
        cleanup_arrays();
        PyErr_SetString(PyExc_TypeError, "reference_data must be a 2-D float64 numpy array and explain_point a float64 numpy array");
        return NULL;
    }

    const double *reference_data = (const double *)PyArray_DATA(reference_data_array);
    const double *explain_data = (const double *)PyArray_DATA(explain_point_array);
    const int n_ref = (int)PyArray_DIM(reference_data_array, 0);
    const int n_features = (int)PyArray_DIM(reference_data_array, 1);

    std::vector<Tree> trees;
    trees.reserve(static_cast<size_t>(num_trees));
    for (Py_ssize_t t = 0; t < num_trees; t++)
    {
        PyArrayObject *vals_arr = convert_item(values_list_obj, t, NPY_FLOAT64);
        PyArrayObject *thr_arr = convert_item(thresholds_list_obj, t, NPY_FLOAT64);
        PyArrayObject *feat_arr = convert_item(features_list_obj, t, NPY_INT64);
        PyArrayObject *cl_arr = convert_item(children_left_list_obj, t, NPY_INT64);
        PyArrayObject *cr_arr = convert_item(children_right_list_obj, t, NPY_INT64);
        PyArrayObject *cm_arr = convert_item(children_missing_list_obj, t, NPY_BOOL);
        PyArrayObject *cat_values_arr = NULL, *cat_start_arr = NULL, *cat_size_arr = NULL;
        if (use_categorical)
        {
            cat_values_arr = convert_item(cat_values_obj, t, NPY_INT64);
            cat_start_arr = convert_item(cat_start_obj, t, NPY_INT64);
            cat_size_arr = convert_item(cat_size_obj, t, NPY_INT64);
        }
        if (!vals_arr || !thr_arr || !feat_arr || !cl_arr || !cr_arr || !cm_arr ||
            (use_categorical && (!cat_values_arr || !cat_start_arr || !cat_size_arr)))
        {
            cleanup_arrays();
            PyErr_SetString(PyExc_TypeError, "Failed to convert tree arrays");
            return NULL;
        }
        trees.push_back(Tree(
            (double *)PyArray_DATA(vals_arr),
            (double *)PyArray_DATA(thr_arr),
            (int64_t *)PyArray_DATA(feat_arr),
            (int64_t *)PyArray_DATA(cl_arr),
            (int64_t *)PyArray_DATA(cr_arr),
            (bool *)PyArray_DATA(cm_arr),
            std::string(decision_type_cptr),
            use_categorical ? (const int64_t *)PyArray_DATA(cat_values_arr) : nullptr,
            use_categorical ? (const int64_t *)PyArray_DATA(cat_start_arr) : nullptr,
            use_categorical ? (const int64_t *)PyArray_DATA(cat_size_arr) : nullptr));
    }

    // Extract custom weight table pointer if provided
    const double *custom_table = nullptr;
    int64_t custom_N = 0, custom_K = 0;
    if (weight_table_obj != Py_None)
    {
        PyArrayObject *weight_table_array = (PyArrayObject *)PyArray_FROM_OTF(weight_table_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
        if (!weight_table_array)
        {
            cleanup_arrays();
            PyErr_SetString(PyExc_TypeError, "weight_table must be a float64 numpy array");
            return NULL;
        }
        arrays_for_decref.push_back(weight_table_array);
        custom_table = (const double *)PyArray_DATA(weight_table_array);
        custom_N = (int64_t)n_features + 1;
        custom_K = (int64_t)max_order + 1;
    }

    int result_size = 0;
    for (int order = 1; order <= max_order; order++)
    {
        result_size += static_cast<int>(inter_weights::binom(n_features, order));
    }
    double *result = new double[result_size]();

    Py_BEGIN_ALLOW_THREADS

    // |E| + |R| never exceeds the longest root-to-leaf path, so the weight tables
    // only need that stride (mirrors the max_e/max_r scan in the flatten kernel).
    int max_depth = 0;
    for (Py_ssize_t t = 0; t < num_trees; t++)
    {
        max_depth = std::max(max_depth, cohort_max_depth(trees[t]));
    }
    max_depth = std::min(max_depth, n_features); // |E|, |R| are feature sets, so also bounded by n_features
    const int table_stride = max_depth + 1;

    double *table_s1 = nullptr;
    double *table_s2 = nullptr;
    double *table_s3 = nullptr;
    if (index_type != IndexType::CUSTOM)
    {
        table_s1 = new double[2 * table_stride * table_stride];
        if (max_order >= 2)
            table_s2 = new double[3 * table_stride * table_stride];
        if (max_order >= 3)
            table_s3 = new double[4 * table_stride * table_stride];
        precompute_weight_tables(index_type, n_features, max_order, table_s1, table_s2, table_s3, table_stride);
    }

    const double inv_scaling = (n_ref > 0) ? 1.0 / (double)n_ref : 0.0;

#pragma omp parallel
    {
        double *local_result = new double[result_size]();
        inter_weights::WeightCache weight_cache = (custom_table != nullptr)
                                                      ? inter_weights::WeightCache((uint64_t)(3 * n_features), custom_table, custom_N, custom_K)
                                                      : inter_weights::WeightCache((uint64_t)(3 * n_features));
        std::vector<int> ref_order(n_ref);
        std::vector<int32_t> leaf_feats(max_depth + 1);
        std::vector<int32_t> leaf_fie(max_depth + 1);
        std::vector<CohortFrame> stack;
        stack.reserve(256);

#pragma omp for schedule(dynamic, 1) nowait
        for (Py_ssize_t t = 0; t < num_trees; t++)
        {
            DISPATCH_INDEX_TYPE(cohort_process_tree, index_type,
                trees[t], reference_data, explain_data, n_ref, n_features,
                inv_scaling, table_s1, table_s2, table_s3, table_stride,
                weight_cache, max_order,
                ref_order, leaf_feats, leaf_fie, stack, local_result);
        }

#pragma omp critical
        {
            for (int k = 0; k < result_size; k++)
                result[k] += local_result[k];
        }
        delete[] local_result;
    }

    delete[] table_s1;
    delete[] table_s2;
    delete[] table_s3;

    Py_END_ALLOW_THREADS

    cleanup_arrays();

    PyObject *output = cohort_result_to_dict(result, n_features, max_order);
    delete[] result;
    return output;
}

// === predict_ensemble_sum ===
// C port of shapiq.tree.base.predict_ensemble: route every row of X through
// every tree (same goes_left semantics as the interaction kernels — NaN routing,
// categorical splits, decision type) and return the per-row sum of leaf values.
// Used for the interventional baseline_value, which was a Python while loop
// over every (row, tree) pair.
static PyObject *predict_ensemble_sum(PyObject *self, PyObject *args)
{
    PyObject *values_list_obj;
    PyObject *thresholds_list_obj;
    PyObject *features_list_obj;
    PyObject *children_left_list_obj;
    PyObject *children_right_list_obj;
    PyObject *children_missing_list_obj;
    PyObject *x_data_obj;
    const char *decision_type_cptr;
    PyObject *cat_values_obj = Py_None;
    PyObject *cat_start_obj = Py_None;
    PyObject *cat_size_obj = Py_None;

    if (!PyArg_ParseTuple(args, "OOOOOOOs|OOO",
                          &values_list_obj, &thresholds_list_obj, &features_list_obj,
                          &children_left_list_obj, &children_right_list_obj,
                          &children_missing_list_obj,
                          &x_data_obj, &decision_type_cptr,
                          &cat_values_obj, &cat_start_obj, &cat_size_obj))
    {
        return NULL;
    }

    const bool use_categorical = (cat_values_obj != Py_None);
    if (!PyList_Check(values_list_obj) || !PyList_Check(thresholds_list_obj) ||
        !PyList_Check(features_list_obj) || !PyList_Check(children_left_list_obj) ||
        !PyList_Check(children_right_list_obj) || !PyList_Check(children_missing_list_obj))
    {
        PyErr_SetString(PyExc_TypeError, "All tree inputs must be lists of numpy arrays");
        return NULL;
    }
    if (use_categorical && (!PyList_Check(cat_values_obj) || !PyList_Check(cat_start_obj) || !PyList_Check(cat_size_obj)))
    {
        PyErr_SetString(PyExc_TypeError, "cat_values, cat_start, and cat_size must be lists of numpy arrays (or None)");
        return NULL;
    }

    Py_ssize_t num_trees = PyList_Size(values_list_obj);
    if (num_trees != PyList_Size(thresholds_list_obj) ||
        num_trees != PyList_Size(features_list_obj) ||
        num_trees != PyList_Size(children_left_list_obj) ||
        num_trees != PyList_Size(children_right_list_obj) ||
        num_trees != PyList_Size(children_missing_list_obj) ||
        (use_categorical && (num_trees != PyList_Size(cat_values_obj) ||
                             num_trees != PyList_Size(cat_start_obj) ||
                             num_trees != PyList_Size(cat_size_obj))))
    {
        PyErr_SetString(PyExc_ValueError, "All tree lists must have the same length");
        return NULL;
    }

    std::vector<PyArrayObject *> arrays_for_decref;
    auto cleanup_arrays = [&arrays_for_decref]()
    {
        for (PyArrayObject *arr : arrays_for_decref)
            Py_XDECREF(arr);
    };
    auto convert_item = [&arrays_for_decref](PyObject *list_obj, Py_ssize_t idx, int np_type) -> PyArrayObject *
    {
        PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
            PyList_GetItem(list_obj, idx), np_type, NPY_ARRAY_IN_ARRAY);
        if (arr)
            arrays_for_decref.push_back(arr);
        return arr;
    };

    PyArrayObject *x_data_array = (PyArrayObject *)PyArray_FROM_OTF(x_data_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x_data_array)
        arrays_for_decref.push_back(x_data_array);
    if (!x_data_array || PyArray_NDIM(x_data_array) != 2)
    {
        cleanup_arrays();
        PyErr_SetString(PyExc_TypeError, "X must be a 2-D float64 numpy array");
        return NULL;
    }

    const double *x_data = (const double *)PyArray_DATA(x_data_array);
    const npy_intp n_rows = PyArray_DIM(x_data_array, 0);
    const npy_intp n_features = PyArray_DIM(x_data_array, 1);

    std::vector<Tree> trees;
    trees.reserve(static_cast<size_t>(num_trees));
    for (Py_ssize_t t = 0; t < num_trees; t++)
    {
        PyArrayObject *vals_arr = convert_item(values_list_obj, t, NPY_FLOAT64);
        PyArrayObject *thr_arr = convert_item(thresholds_list_obj, t, NPY_FLOAT64);
        PyArrayObject *feat_arr = convert_item(features_list_obj, t, NPY_INT64);
        PyArrayObject *cl_arr = convert_item(children_left_list_obj, t, NPY_INT64);
        PyArrayObject *cr_arr = convert_item(children_right_list_obj, t, NPY_INT64);
        PyArrayObject *cm_arr = convert_item(children_missing_list_obj, t, NPY_BOOL);
        PyArrayObject *cat_values_arr = NULL, *cat_start_arr = NULL, *cat_size_arr = NULL;
        if (use_categorical)
        {
            cat_values_arr = convert_item(cat_values_obj, t, NPY_INT64);
            cat_start_arr = convert_item(cat_start_obj, t, NPY_INT64);
            cat_size_arr = convert_item(cat_size_obj, t, NPY_INT64);
        }
        if (!vals_arr || !thr_arr || !feat_arr || !cl_arr || !cr_arr || !cm_arr ||
            (use_categorical && (!cat_values_arr || !cat_start_arr || !cat_size_arr)))
        {
            cleanup_arrays();
            PyErr_SetString(PyExc_TypeError, "Failed to convert tree arrays");
            return NULL;
        }
        trees.push_back(Tree(
            (double *)PyArray_DATA(vals_arr),
            (double *)PyArray_DATA(thr_arr),
            (int64_t *)PyArray_DATA(feat_arr),
            (int64_t *)PyArray_DATA(cl_arr),
            (int64_t *)PyArray_DATA(cr_arr),
            (bool *)PyArray_DATA(cm_arr),
            std::string(decision_type_cptr),
            use_categorical ? (const int64_t *)PyArray_DATA(cat_values_arr) : nullptr,
            use_categorical ? (const int64_t *)PyArray_DATA(cat_start_arr) : nullptr,
            use_categorical ? (const int64_t *)PyArray_DATA(cat_size_arr) : nullptr));
    }

    npy_intp out_dim = n_rows;
    PyObject *out_array = PyArray_SimpleNew(1, &out_dim, NPY_FLOAT64);
    if (!out_array)
    {
        cleanup_arrays();
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array");
        return NULL;
    }
    double *out = (double *)PyArray_DATA((PyArrayObject *)out_array);

    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n_rows; i++)
    {
        const double *row = x_data + i * n_features;
        double total = 0.0;
        for (size_t t = 0; t < trees.size(); t++)
        {
            Tree &tree = trees[t];
            int64_t node = 0;
            while (tree.children_left[node] != tree.children_right[node])
            {
                const int64_t feature = tree.features[node];
                node = tree.goes_left(row[feature], node)
                           ? tree.children_left[node]
                           : tree.children_right[node];
            }
            total += tree.leaf_predictions[node];
        }
        out[i] = total;
    }
    Py_END_ALLOW_THREADS

    cleanup_arrays();
    return out_array;
}
