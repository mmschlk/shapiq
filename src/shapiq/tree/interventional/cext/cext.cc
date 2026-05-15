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
static PyObject *compute_interactions(PyObject *self, PyObject *args);
static PyObject *compute_interactions_batched(PyObject *self, PyObject *args);
static PyObject *compute_interactions_batched_sparse(PyObject *self, PyObject *args);
static PyObject *compute_interactions_flatten(PyObject *self, PyObject *args);
static PyObject *compute_interactions_sparse(PyObject *self, PyObject *args);
static PyObject *preprocess_boolean_trees(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"compute_interactions", compute_interactions, METH_VARARGS, "Compute feature interactions using the interventional algorithm."},
    {"compute_interactions_batched", compute_interactions_batched, METH_VARARGS, "Compute feature interactions in batches using the interventional algorithm."},
    {"compute_interactions_batched_sparse", compute_interactions_batched_sparse, METH_VARARGS, "Compute sparse feature interactions in batches using the interventional algorithm."},
    {"compute_interactions_flatten", compute_interactions_flatten, METH_VARARGS, "Compute feature interactions with flattened input."},
    {"compute_interactions_sparse", compute_interactions_sparse, METH_VARARGS, "Compute sparse feature interactions using the interventional algorithm."},
    {"preprocess_boolean_trees", preprocess_boolean_trees, METH_VARARGS, "Preprocess boolean trees: DFS traversal to produce flattened E/R arrays."},
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

static PyObject *compute_interactions(PyObject *self, PyObject *args)
{

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

    // 2. Step parse input arguements
    if (!PyArg_ParseTuple(args, "OOOOOOOOssii", &leaf_predictions_obj, &thresholds_obj, &features_obj, &children_left_obj, &children_right_obj, &children_missing_obj, &reference_data_obj, &explain_data_obj, &decision_type_cptr, &index_cptr, &max_order, &verbose))
    {
        return NULL; // Return NULL if argument parsing fails
    }
    // Throw an error if the input data is not in the expected format
    if (!PyArray_Check(leaf_predictions_obj) || !PyArray_Check(thresholds_obj) || !PyArray_Check(features_obj) || !PyArray_Check(children_left_obj) || !PyArray_Check(children_right_obj) || !PyArray_Check(children_missing_obj) || !PyArray_Check(reference_data_obj) || !PyArray_Check(explain_data_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Input data must be numpy arrays");
        return NULL;
    }
    // 3. Step convert input to numpy arrays. The extraction uses the flexible PyArray_FROM_OTF function, which allows for various input types and ensures that the data is in the correct format for processing.
    // The second arguement specifies the desired data type (e.g., NPY_FLOAT32 for 32-bit floating-point numbers, NPY_INT for integers), while the third argument specifies the requirements for the array (e.g., NPY_ARRAY_IN_ARRAY for read-only access).
    PyArrayObject *leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_predictions_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject *)PyArray_FROM_OTF(thresholds_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject *)PyArray_FROM_OTF(features_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_left_array = (PyArrayObject *)PyArray_FROM_OTF(children_left_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject *)PyArray_FROM_OTF(children_right_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_missing_array = (PyArrayObject *)PyArray_FROM_OTF(children_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);

    PyArrayObject *reference_data_array = (PyArrayObject *)PyArray_FROM_OTF(reference_data_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *explain_data_array = (PyArrayObject *)PyArray_FROM_OTF(explain_data_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    string decision_type = string(decision_type_cptr);
    string index = string(index_cptr);
    // Check if any of the array conversions failed and handle the error appropriately
    // Here we use Py_XDECREF, as Py_CLEAR is not necessary since we are returning NULL immediately after. This ensures that any successfully created arrays are properly decremented, preventing memory leaks.
    if (!leaf_predictions_array || !thresholds_array || !features_array || !children_left_array || !children_right_array || !children_missing_array || !reference_data_array || !explain_data_array)
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(features_array);
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_missing_array);
        Py_XDECREF(reference_data_array);
        Py_XDECREF(explain_data_array);
        return NULL; // Return NULL if any array conversion fails
    }

    if (verbose > 0)
    {
        cout << "Successfully parsed input arguments and converted to numpy arrays." << endl;
        cout << "Leaf predictions array shape: " << leaf_predictions_array->dimensions[0] << endl;
        cout << "Thresholds array shape: " << thresholds_array->dimensions[0] << endl;
        cout << "Features array shape: " << features_array->dimensions[0] << endl;
        cout << "Children left array shape: " << children_left_array->dimensions[0] << endl;
        cout << "Children right array shape: " << children_right_array->dimensions[0] << endl;
        cout << "Children missing array shape: " << children_missing_array->dimensions[0] << endl;
        cout << "Reference data array shape: " << reference_data_array->dimensions[0] << " x " << reference_data_array->dimensions[1] << endl;
        cout << "Explain data array shape: " << explain_data_array->dimensions[0] << endl;
        cout << "Decision type: " << decision_type << endl;
        cout << "Index: " << index << endl;
        cout << "Max order: " << max_order << endl;
    }
    // 4. Step extract C-type pointers from the numpy arrays.
    float *leaf_predictions = (float *)PyArray_DATA(leaf_predictions_array);
    float *thresholds = (float *)PyArray_DATA(thresholds_array);
    int64_t *features = (int64_t *)PyArray_DATA(features_array);
    int64_t *children_left = (int64_t *)PyArray_DATA(children_left_array);
    int64_t *children_right = (int64_t *)PyArray_DATA(children_right_array);
    bool *children_missing = (bool *)PyArray_DATA(children_missing_array);

    float *reference_data = (float *)PyArray_DATA(reference_data_array);
    float *explain_data = (float *)PyArray_DATA(explain_data_array);

    if (verbose > 0)
    {
        cout << "Successfully extracted C-type pointers from numpy arrays." << endl;
    }

    // 5. Step create Tree object
    Tree tree = Tree(leaf_predictions, thresholds, features, children_left, children_right, children_missing, decision_type);
    int n_reference_samples = static_cast<int>(reference_data_array->dimensions[0]);
    int n_features = static_cast<int>(reference_data_array->dimensions[1]);

    // Calculate result array size based on max_order
    // Uses compact representation: sum of binomial coefficients
    // For max_order=1: C(n,1) = n (main effects only)
    // For max_order=2: C(n,1) + C(n,2) = n + n*(n-1)/2 (main effects + pairwise)
    int result_size = 0;
    for (int order = 1; order <= max_order; order++)
    {
        result_size += static_cast<int>(inter_weights::binom(n_features, order));
    }

    double *result = new double[result_size];
    for (int i = 0; i < result_size; i++)
    {
        result[i] = 0.0;
    }

    IndexType index_type;
    if (!parse_index_type(index, index_type))
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(features_array);
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_missing_array);
        Py_XDECREF(reference_data_array);
        Py_XDECREF(explain_data_array);
        PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + index).c_str());
        return NULL; // Return NULL if index type is unsupported
    }

    // 4. Step run interventional algorithm.
    inter_weights::WeightCache weight_cache(3 * n_features); // Cache for weights to avoid redundant calculations. The maximum number of features in A or B is num_features, so we can have at most 2*num_features for a and b. We multiply by 3 to be safe and avoid collisions.
    if (verbose > 0)
    {
        cout << "Starting computation of interactions for " << n_reference_samples << " reference samples and " << n_features << " features." << endl;
    }
    for (int i = 0; i < n_reference_samples; i++)
    {
        // Get pointer to the current reference sample
        float *reference_sample = reference_data + i * n_features;
        // Compute interactions for the current reference sample and the explain sample
        if (verbose > 0)
        {
            cout << "Computing interactions for reference sample " << i + 1 << "/" << n_reference_samples << endl;
        }
        algorithms::compute_interactions(tree, result, weight_cache, reference_sample, explain_data, n_features, index_type, max_order, verbose);
    }
    for (int i = 0; i < result_size; i++)
    {
        result[i] /= n_reference_samples; // Average interactions over all reference samples
    }

    // 5. Step convert output to numpy array
    npy_intp dims[1] = {(npy_intp)result_size};
    PyObject *output = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    memcpy(PyArray_DATA((PyArrayObject *)output), result, result_size * sizeof(double));
    delete[] result; // Free memory allocated for the result array
    if (verbose > 0)
    {
        cout << "Successfully computed interactions and converted output to numpy array." << endl;
    }
    // 6. Step clean up and return output
    Py_XDECREF(leaf_predictions_array);
    Py_XDECREF(thresholds_array);
    Py_XDECREF(features_array);
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(children_missing_array);
    Py_XDECREF(reference_data_array);
    Py_XDECREF(explain_data_array);
    if (verbose > 0)
    {
        cout << "Successfully computed interactions and converted output to numpy array." << endl;
    }
    return output; // Return the output numpy array containing the computed interactions
}

static PyObject *compute_interactions_batched(PyObject *self, PyObject *args)
{
    /**
     * Batched version of compute_interactions.
     * Now each input in args is a list of numpy arrays, where each array corresponds to a different tree.
     * The function computes interaction for each tree in parallel and returns a numpy array containing the average interactions across all trees.
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

    // 2. Step parse input arguements
    if (!PyArg_ParseTuple(args, "OOOOOOOOssii", &leaf_predictions_obj, &thresholds_obj, &features_obj, &children_left_obj, &children_right_obj, &children_missing_obj, &reference_data_obj, &explain_data_obj, &decision_type_cptr, &index_cptr, &max_order, &verbose))
    {
        return NULL; // Return NULL if argument parsing fails
    }
    // Throw an error if the input data is not in the expected format
    if (!PyList_Check(leaf_predictions_obj) || !PyList_Check(thresholds_obj) || !PyList_Check(features_obj) || !PyList_Check(children_left_obj) || !PyList_Check(children_right_obj) || !PyList_Check(children_missing_obj) || !PyArray_Check(reference_data_obj) || !PyArray_Check(explain_data_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Input data must be numpy arrays");
        return NULL;
    }

    // 3. Step convert the lists of numpy arrays to a vector of Tree objects.
    // We assume that the input lists of numpy arrays are of the same length, and each index corresponds to a different tree.
    PyObject *iterator_leaf = PyObject_GetIter(leaf_predictions_obj);
    PyObject *iterator_thresholds = PyObject_GetIter(thresholds_obj);
    PyObject *iterator_features = PyObject_GetIter(features_obj);
    PyObject *iterator_children_left = PyObject_GetIter(children_left_obj);
    PyObject *iterator_children_right = PyObject_GetIter(children_right_obj);
    PyObject *iterator_children_missing = PyObject_GetIter(children_missing_obj);
    string index = string(index_cptr);

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
    // Iterate through the lists and create Tree objects for each tree
    PyArrayObject *leaf_predictions_array, *thresholds_array, *features_array, *children_left_array, *children_right_array, *children_missing_array;
    PyObject *leaf_pred_iter, *thresholds_iter, *features_iter, *children_left_iter, *children_right_iter, *children_missing_iter;
    std::vector<Tree> trees;
    // We also need to keep track of the PyArrayObjects we create for each tree, so we can properly decref them later to avoid memory leaks. We can store them in a vector of tuples.
    std::vector<std::tuple<PyArrayObject *, PyArrayObject *, PyArrayObject *, PyArrayObject *, PyArrayObject *, PyArrayObject *>> arrays_for_decref;
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
            // If any of the iterators is exhausted, we break the loop. We also check if all iterators are exhausted to ensure that the input lists are of the same length.
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
                PyErr_SetString(PyExc_ValueError, "Input lists of numpy arrays must be of the same length");
                return NULL;
            }
            break; // All iterators are exhausted, we can exit the loop
        }
        leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_pred_iter, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
        thresholds_array = (PyArrayObject *)PyArray_FROM_OTF(thresholds_iter, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
        features_array = (PyArrayObject *)PyArray_FROM_OTF(features_iter, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        children_left_array = (PyArrayObject *)PyArray_FROM_OTF(children_left_iter, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        children_right_array = (PyArrayObject *)PyArray_FROM_OTF(children_right_iter, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        children_missing_array = (PyArrayObject *)PyArray_FROM_OTF(children_missing_iter, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
        if (!leaf_predictions_array || !thresholds_array || !features_array || !children_left_array || !children_right_array || !children_missing_array)
        {
            Py_XDECREF(leaf_predictions_array);
            Py_XDECREF(thresholds_array);
            Py_XDECREF(features_array);
            Py_XDECREF(children_left_array);
            Py_XDECREF(children_right_array);
            Py_XDECREF(children_missing_array);
            // DECREF any previously created arrays for the trees we have already processed, as we are going to return NULL due to the error. This ensures that we don't leak memory for those arrays.
            for (auto &arrays_tuple : arrays_for_decref)
            {
                Py_XDECREF(std::get<0>(arrays_tuple));
                Py_XDECREF(std::get<1>(arrays_tuple));
                Py_XDECREF(std::get<2>(arrays_tuple));
                Py_XDECREF(std::get<3>(arrays_tuple));
                Py_XDECREF(std::get<4>(arrays_tuple));
                Py_XDECREF(std::get<5>(arrays_tuple));
            }
            PyErr_SetString(PyExc_TypeError, "Each tree's parameters must be numpy arrays");
            return NULL;
        }
        // DECREF the iterators for the current tree parameters, as we have already converted them to numpy arrays and extracted the data pointers. This ensures that we don't leak memory for the iterators.
        Py_XDECREF(leaf_pred_iter);
        Py_XDECREF(thresholds_iter);
        Py_XDECREF(features_iter);
        Py_XDECREF(children_left_iter);
        Py_XDECREF(children_right_iter);
        Py_XDECREF(children_missing_iter);

        trees.push_back(Tree(
            (float *)PyArray_DATA(leaf_predictions_array),
            (float *)PyArray_DATA(thresholds_array),
            (int64_t *)PyArray_DATA(features_array),
            (int64_t *)PyArray_DATA(children_left_array),
            (int64_t *)PyArray_DATA(children_right_array),
            (bool *)PyArray_DATA(children_missing_array),
            std::string(decision_type_cptr)));
        arrays_for_decref.push_back(std::make_tuple(leaf_predictions_array, thresholds_array, features_array, children_left_array, children_right_array, children_missing_array));
        num_trees++;
    }

    // Clean up iterators
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

    if (verbose > 0)
    {
        cout << "Successfully parsed input arguments and converted to Tree objects." << endl;
        cout << "Number of trees: " << num_trees << endl;
    }

    // Extract reference and explain data as before
    PyArrayObject *reference_data_array = (PyArrayObject *)PyArray_FROM_OTF(reference_data_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *explain_data_array = (PyArrayObject *)PyArray_FROM_OTF(explain_data_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (!reference_data_array || !explain_data_array)
    {
        Py_XDECREF(reference_data_array);
        Py_XDECREF(explain_data_array);
        PyErr_SetString(PyExc_TypeError, "Reference and explain data must be numpy arrays");
        return NULL;
    }
    float *reference_data = (float *)PyArray_DATA(reference_data_array);
    float *explain_data = (float *)PyArray_DATA(explain_data_array);
    if (verbose > 0)
    {
        cout << "Successfully extracted C-type pointers from numpy arrays." << endl;
    }
    int n_reference_samples = static_cast<int>(reference_data_array->dimensions[0]);
    int n_features = static_cast<int>(reference_data_array->dimensions[1]);

    // Calculate result array size based on max_order
    // Uses compact representation: sum of binomial coefficients
    // For max_order=1: C(n,1) = n (main effects only)
    // For max_order=2: C(n,1) + C(n,2) = n + n*(n-1)/2 (main effects + pairwise)
    int result_size = 0;
    for (int order = 1; order <= max_order; order++)
    {
        result_size += static_cast<int>(inter_weights::binom(n_features, order));
    }

    double *result = new double[result_size];
    for (int i = 0; i < result_size; i++)
    {
        result[i] = 0.0;
    }

    IndexType index_type;
    if (!parse_index_type(index, index_type))
    {
        Py_XDECREF(reference_data_array);
        Py_XDECREF(explain_data_array);
        for (const auto &arr_tuple : arrays_for_decref)
        {
            Py_XDECREF(std::get<0>(arr_tuple)); // leaf_predictions_array
            Py_XDECREF(std::get<1>(arr_tuple)); // thresholds_array
            Py_XDECREF(std::get<2>(arr_tuple)); // features_array
            Py_XDECREF(std::get<3>(arr_tuple)); // children_left_array
            Py_XDECREF(std::get<4>(arr_tuple)); // children_right_array
            Py_XDECREF(std::get<5>(arr_tuple)); // children_missing_array
        }
        PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + index).c_str());
        return NULL; // Return NULL if index type is unsupported
    }

    // 4. Step run interventional algorithm in parallel for each tree and average results
    inter_weights::WeightCache weight_cache(3 * n_features); // Cache for weights to avoid redundant calculations. The maximum number of features in A or B is num_features, so we can have at most 2*num_features for a and b. We multiply by 3 to be safe and avoid collisions.
    // Sequential
    for (int t = 0; t < num_trees; t++)
    {
        for (int i = 0; i < n_reference_samples; i++)
        {
            float *reference_sample = reference_data + i * n_features;
            algorithms::compute_interactions(
                trees[t],
                result, // global result array
                weight_cache,
                reference_sample,
                explain_data,
                n_features,
                index_type,
                max_order,
                verbose);
        }
    }

    //     Py_BEGIN_ALLOW_THREADS
    // #pragma omp parallel // Declare parallel region for OpenMP.
    //     {
    //         // Each thread gets its own local buffer to accumulate results
    //         std::vector<double> local(result_size, 0.0);
    //         inter_weights::WeightCache weight_cache(3 * n_features);

    // #pragma omp for nowait // Distribute the loop iterations over the threads without an implicit barrier at the end, allowing threads to start processing as soon as they finish their assigned iterations.
    //         for (int t = 0; t < num_trees; t++)
    //         {
    //             for (int i = 0; i < n_reference_samples; i++)
    //             {
    //                 float *reference_sample = reference_data + i * n_features;
    //                 algorithms::compute_interactions(
    //                     trees[t],
    //                     local.data(), // thread-local buffer
    //                     weight_cache,
    //                     reference_sample,
    //                     explain_data,
    //                     n_features,
    //                     index_type,
    //                     max_order,
    //                     verbose);
    //             }
    //         }

    // #pragma omp critical // Merge results from each thread into the global result array in a critical section
    //         {
    //             for (int k = 0; k < result_size; k++)
    //             {
    //                 result[k] += local[k];
    //             }
    //         }
    //     }
    //     Py_END_ALLOW_THREADS

    for (int i = 0; i < result_size; i++)
    {
        result[i] /= (n_reference_samples);
    }
    // 5. Step convert output to numpy array
    npy_intp output_dims[1] = {(npy_intp)result_size};
    PyObject *output = PyArray_SimpleNew(1, output_dims, NPY_FLOAT64);
    memcpy(PyArray_DATA((PyArrayObject *)output), result, result_size * sizeof(double));
    delete[] result; // Free memory allocated for the result array

    // 6. Step clean up and return output
    Py_XDECREF(reference_data_array);
    Py_XDECREF(explain_data_array);
    for (const auto &arr_tuple : arrays_for_decref)
    {
        Py_XDECREF(std::get<0>(arr_tuple)); // leaf_predictions_array
        Py_XDECREF(std::get<1>(arr_tuple)); // thresholds_array
        Py_XDECREF(std::get<2>(arr_tuple)); // features_array
        Py_XDECREF(std::get<3>(arr_tuple)); // children_left_array
        Py_XDECREF(std::get<4>(arr_tuple)); // children_right_array
        Py_XDECREF(std::get<5>(arr_tuple)); // children_missing_array
    }
    if (verbose > 0)
    {
        cout << "Successfully computed batched interactions and converted output to numpy array." << endl;
    }
    return output; // Return the output numpy array containing the computed interactions
}

static PyObject *compute_interactions_batched_sparse(PyObject *self, PyObject *args)
{
    /**
     * This function is an extension of the batched version of compute_interactions, which computes interactions in parallel for each tree and returns a sparse representation of the interactions as a dictionary mapping feature subsets to interaction values.
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

    if (!PyArg_ParseTuple(args, "OOOOOOOOssii|O", &leaf_predictions_obj, &thresholds_obj, &features_obj, &children_left_obj, &children_right_obj, &children_missing_obj, &reference_data_obj, &explain_data_obj, &decision_type_cptr, &index_cptr, &max_order, &verbose, &weight_table_obj))
    {
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
                PyErr_SetString(PyExc_ValueError, "Input lists of numpy arrays must be of the same length");
                return NULL;
            }
            break;
        }

        leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_pred_iter, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
        thresholds_array = (PyArrayObject *)PyArray_FROM_OTF(thresholds_iter, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
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
            PyErr_SetString(PyExc_TypeError, "Each tree's parameters must be numpy arrays");
            return NULL;
        }

        trees.push_back(Tree(
            (float *)PyArray_DATA(leaf_predictions_array),
            (float *)PyArray_DATA(thresholds_array),
            (int64_t *)PyArray_DATA(features_array),
            (int64_t *)PyArray_DATA(children_left_array),
            (int64_t *)PyArray_DATA(children_right_array),
            (bool *)PyArray_DATA(children_missing_array),
            std::string(decision_type_cptr)));
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

    PyArrayObject *reference_data_array = (PyArrayObject *)PyArray_FROM_OTF(reference_data_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *explain_data_array = (PyArrayObject *)PyArray_FROM_OTF(explain_data_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
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
        PyErr_SetString(PyExc_TypeError, "Reference and explain data must be numpy arrays");
        return NULL;
    }

    float *reference_data = (float *)PyArray_DATA(reference_data_array);
    float *explain_data = (float *)PyArray_DATA(explain_data_array);
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
                float *reference_sample = reference_data + i * n_features;
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

    return output;
}

// === Optimized helpers for compute_interactions_flatten ===

// Compute signed weight matching the original per-index logic, for table precomputation.
static inline double compute_signed_weight_for_table(
    IndexType index_type, int n_features, int e, int r,
    int s_cap_e, int s_cap_r, int s, int max_order)
{
    int sign = (s_cap_r % 2 == 0) ? 1 : -1;
    switch (index_type)
    {
    case IndexType::SII:
        return sign * inter_weights::shapley_weight(n_features, e, r, s_cap_e, s_cap_r, s, max_order);
    case IndexType::BII:
        return sign * inter_weights::banzhaf_weight(n_features, e, r, s_cap_e, s_cap_r, s, max_order);
    case IndexType::CHII:
        return sign * inter_weights::chaining_weight(n_features, e, r, s_cap_e, s_cap_r, s, max_order);
    case IndexType::FBII:
        return inter_weights::fbii_weight(n_features, e, r, s_cap_e, s_cap_r, s, max_order);
    case IndexType::FSII:
        return inter_weights::fsii_weight(n_features, e, r, s_cap_e, s_cap_r, s, max_order);
    default:
        return sign * inter_weights::general_weight(n_features, e, r, s_cap_e, s_cap_r, s, max_order, index_type);
    }
}

// Precompute weight lookup tables.
// table_s1: indexed by [s_cap_e * stride^2 + e * stride + r], s_cap_e in {0,1}
// table_s2: indexed by [s_cap_e_combined * stride^2 + e * stride + r], s_cap_e_combined in {0,1,2}
// table_s3: indexed by [s_cap_e_combined * stride^2 + e * stride + r], s_cap_e_combined in {0,1,2,3}
static void precompute_weight_tables(
    IndexType index_type, int n_features, int max_order,
    float *table_s1, float *table_s2, float *table_s3, int table_stride)
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
                double w = compute_signed_weight_for_table(
                    index_type, n_features, e, r, s_cap_e, s_cap_r, 1, max_order);
                table_s1[s_cap_e * table_stride * table_stride + e * table_stride + r] = (float)w;
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
                    double w = compute_signed_weight_for_table(
                        index_type, n_features, e, r, s_cap_e_c, s_cap_r_c, 2, max_order);
                    table_s2[s_cap_e_c * table_stride * table_stride + e * table_stride + r] = (float)w;
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
                    double w = compute_signed_weight_for_table(
                        index_type, n_features, e, r, s_cap_e_c, s_cap_r_c, 3, max_order);
                    table_s3[s_cap_e_c * table_stride * table_stride + e * table_stride + r] = (float)w;
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
    const float *__restrict__ leaf_pred,
    const int32_t *__restrict__ feat32,
    const int32_t *__restrict__ e32,
    const int32_t *__restrict__ r32,
    const int32_t *__restrict__ fie32,
    const float *__restrict__ e_f32,
    const float *__restrict__ r_f32,
    const float *__restrict__ fie_f32,
    int n_iterations,
    int n_features,
    float inv_scaling,
    const float *__restrict__ table_s1,
    int table_stride,
    inter_weights::WeightCache &weight_cache,
    int max_order,
    double *__restrict__ result)
{
    float *contrib = new float[n_iterations];

    if constexpr (IT == IndexType::BII)
    {
        for (int i = 0; i < n_iterations; i++)
        {
            float sign = 1.0f - 2.0f * (1.0f - fie_f32[i]);
            float w = exp2f(-(e_f32[i] + r_f32[i] - 1.0f));
            contrib[i] = leaf_pred[i] * sign * w * inv_scaling;
        }
    }
    else if constexpr (IT == IndexType::CUSTOM)
    {
        for (int i = 0; i < n_iterations; i++)
        {
            float w = (float)weight_cache.get_weight(
                n_features, e32[i], r32[i], fie32[i], 1 - fie32[i], 1, IT, max_order);
            contrib[i] = leaf_pred[i] * w * inv_scaling;
        }
    }
    else
    {
        float *weights = new float[n_iterations];
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
    const float *__restrict__ leaf_pred,
    const int32_t *__restrict__ feat32,
    const int32_t *__restrict__ e32,
    const int32_t *__restrict__ r32,
    const int32_t *__restrict__ fie32,
    const int32_t *__restrict__ lid32,
    const float *__restrict__ e_f32,
    const float *__restrict__ r_f32,
    const float *__restrict__ fie_f32,
    int n_iterations,
    int n_features,
    float inv_scaling,
    const float *__restrict__ table_s1,
    const float *__restrict__ table_s2,
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
                float w1;
                if constexpr (IT == IndexType::BII)
                {
                    float sign = 1.0f - 2.0f * (1.0f - fie_f32[i]);
                    w1 = sign * exp2f(-(e_f32[i] + r_f32[i] - 1.0f));
                }
                else if constexpr (IT == IndexType::CUSTOM)
                {
                    w1 = (float)weight_cache.get_weight(
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
                    float w2;
                    if constexpr (IT == IndexType::BII)
                    {
                        int s_cap_r_c = 2 - s_cap_e_c;
                        float sign_c = (s_cap_r_c % 2 == 0) ? 1.0f : -1.0f;
                        w2 = sign_c * exp2f(-(e_f32[i] + r_f32[i] - 2.0f));
                    }
                    else if constexpr (IT == IndexType::CUSTOM)
                    {
                        int s_cap_r_c = 2 - s_cap_e_c;
                        w2 = (float)weight_cache.get_weight(
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
    const float *__restrict__ leaf_pred,
    const int32_t *__restrict__ feat32,
    const int32_t *__restrict__ e32,
    const int32_t *__restrict__ r32,
    const int32_t *__restrict__ fie32,
    const int32_t *__restrict__ lid32,
    const float *__restrict__ e_f32,
    const float *__restrict__ r_f32,
    const float *__restrict__ fie_f32,
    int n_iterations,
    int n_features,
    float inv_scaling,
    const float *__restrict__ table_s1,
    const float *__restrict__ table_s2,
    const float *__restrict__ table_s3,
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
                float lp = leaf_pred[i] * inv_scaling;

                // Order-1 contribution
                float w1;
                if constexpr (IT == IndexType::BII)
                {
                    float sign = 1.0f - 2.0f * (1.0f - fie_f32[i]);
                    w1 = sign * exp2f(-(e_f32[i] + r_f32[i] - 1.0f));
                }
                else if constexpr (IT == IndexType::CUSTOM)
                {
                    w1 = (float)weight_cache.get_weight(
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
                    float w2;
                    if constexpr (IT == IndexType::BII)
                    {
                        int s_cap_r_c = 2 - s_cap_e_2;
                        float sign_c = (s_cap_r_c % 2 == 0) ? 1.0f : -1.0f;
                        w2 = sign_c * exp2f(-(e_f32[i] + r_f32[i] - 2.0f));
                    }
                    else if constexpr (IT == IndexType::CUSTOM)
                    {
                        int s_cap_r_c = 2 - s_cap_e_2;
                        w2 = (float)weight_cache.get_weight(
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
                        float w3;
                        if constexpr (IT == IndexType::BII)
                        {
                            int s_cap_r_3 = 3 - s_cap_e_3;
                            float sign_3 = (s_cap_r_3 % 2 == 0) ? 1.0f : -1.0f;
                            w3 = sign_3 * exp2f(-(e_f32[i] + r_f32[i] - 3.0f));
                        }
                        else if constexpr (IT == IndexType::CUSTOM)
                        {
                            int s_cap_r_3 = 3 - s_cap_e_3;
                            w3 = (float)weight_cache.get_weight(
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
    float scaling_factor = 1.0;
    // Optional custom weight table (None when not used)
    PyObject *weight_table_obj = Py_None;
    IndexType index_type;
    if (!PyArg_ParseTuple(args, "OOOOOOsiiiiif|O", &leaf_predictions_obj, &features_obj, &e_sizes, &r_sizes, &feature_in_e_obj, &leaf_id, &index_cptr, &n_iterations, &n_features, &e_length, &max_order, &verbose, &scaling_factor, &weight_table_obj))
    {
        return NULL;
    }
    if (!PyArray_Check(leaf_predictions_obj) || !PyArray_Check(features_obj) || !PyArray_Check(e_sizes) || !PyArray_Check(r_sizes) || !PyArray_Check(feature_in_e_obj) || !PyArray_Check(leaf_id))
    {
        PyErr_SetString(PyExc_TypeError, "Input data must be numpy arrays");
        return NULL;
    }
    PyArrayObject *leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_predictions_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
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
    float *leaf_predictions = (float *)PyArray_DATA(leaf_predictions_array);
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
        // --- Phase 0: Convert int64 inputs to int32 + float32 for SIMD-friendly processing ---
        int32_t *feat32 = new int32_t[n_iterations];
        int32_t *e32 = new int32_t[n_iterations];
        int32_t *r32 = new int32_t[n_iterations];
        int32_t *fie32 = new int32_t[n_iterations];
        int32_t *lid32 = new int32_t[n_iterations];
        float *e_f32 = new float[n_iterations];
        float *r_f32 = new float[n_iterations];
        float *fie_f32 = new float[n_iterations];
        for (int i = 0; i < n_iterations; i++)
        {
            feat32[i] = (int32_t)features[i];
            e32[i] = (int32_t)e_sizes_data[i];
            r32[i] = (int32_t)r_sizes_data[i];
            fie32[i] = (int32_t)feature_in_e_data[i];
            lid32[i] = (int32_t)leaf_id_data[i];
            e_f32[i] = (float)e_sizes_data[i];
            r_f32[i] = (float)r_sizes_data[i];
            fie_f32[i] = (float)feature_in_e_data[i];
        }

        float inv_scaling = 1.0f / scaling_factor;

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

        float *table_s1 = nullptr;
        float *table_s2 = nullptr;
        float *table_s3 = nullptr;
        if (index_type != IndexType::CUSTOM)
        {
            table_s1 = new float[2 * table_stride * table_stride];
            if (max_order >= 2)
                table_s2 = new float[3 * table_stride * table_stride];
            if (max_order >= 3)
                table_s3 = new float[4 * table_stride * table_stride];
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
                e_f32, r_f32, fie_f32,
                n_iterations, n_features, inv_scaling,
                table_s1, table_stride, weight_cache, max_order, result);
        }
        else if (max_order == 2)
        {
            DISPATCH_INDEX_TYPE(compute_order2_leafparallel, index_type,
                leaf_predictions, feat32, e32, r32, fie32, lid32,
                e_f32, r_f32, fie_f32,
                n_iterations, n_features, inv_scaling,
                table_s1, table_s2, table_stride,
                weight_cache, max_order, result_size, result);
        }
        else if (max_order == 3)
        {
            DISPATCH_INDEX_TYPE(compute_order3_leafparallel, index_type,
                leaf_predictions, feat32, e32, r32, fie32, lid32,
                e_f32, r_f32, fie_f32,
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
        delete[] e_f32;
        delete[] r_f32;
        delete[] fie_f32;
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
    std::vector<float> leaf_vals_out;
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
            PyList_GetItem(values_list_obj, t), NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
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

        float *values = (float *)PyArray_DATA(vals_arr);
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
                float leaf_val = values[node_id];
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
    PyObject *np_leaf_vals = PyArray_SimpleNew(1, &n_total, NPY_FLOAT32);
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
        memcpy(PyArray_DATA((PyArrayObject *)np_leaf_vals), leaf_vals_out.data(), n_total * sizeof(float));
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

static PyObject *compute_interactions_sparse(PyObject *self, PyObject *args)
{
    /**
     * Computes interactions for a single tree using a sparse representation of the tree structure, which is more efficient for  max_order bigger than 2.
     * The function should be used if max_order is bigger than 2.
     * The funcion has the following input parameters( in exact order):
     * - leaf_predictions: A numpy array containing the predictions at the leaf nodes of a decision tree.
     * - e: A numpy array containing the feature indices in the "e" set for each node in the flattened tree representation. Important: The array should have the format (n_samples,n_features) and features not used in the corresponding set should be filled with -1. This is to ensure that the function can efficiently process the features in E and R for each node without needing to check for variable-length lists or additional size information.
     * - r: A numpy array containing the feature indices in the "r" set for each node in the flattened tree representation. Important: The array should have the format (n_samples,n_features) and features not used in the corresponding set should be filled with -1. This is to ensure that the function can efficiently process the features in E and R for each node without needing to check for variable-length lists or additional size information.
     * - e_sizes A numpy array containing the sizes of the subsets of features taken according to the point explained ("e") for each node in the flattened tree representation.
     * - r_sizes A numpy array containing the sizes of the subsets of features taken according to the reference point ("r") set for each node in the flattened tree representation.
     * - index: A string indicating the type of index to compute (e.g., "shapley", "banzhaf", "chaining", "fbii", or "custom"). This determines the weighting scheme used in the interaction computation.
     * - n_features: An integer representing the total number of features in the dataset. This is used to determine the size of the weight cache and to compute the weights for interactions.
     * - max_order: An integer representing the maximum order of interactions to compute. This determines how many features can be involved in an interaction (e.g., max_order=2 means only pairwise interactions will be computed).
     *
     */
    PyObject *leaf_predictions_obj;
    PyObject *e_obj;
    PyObject *r_obj;
    PyObject *e_sizes;
    PyObject *r_sizes;
    int n_features;
    int max_order;
    const char *index_cptr;

    if (!PyArg_ParseTuple(args, "OOOOOsii", &leaf_predictions_obj, &e_obj, &r_obj, &e_sizes, &r_sizes, &index_cptr, &n_features, &max_order))
    {
        return NULL;
    }
    if (!PyArray_Check(leaf_predictions_obj) || !PyArray_Check(e_obj) || !PyArray_Check(r_obj) || !PyArray_Check(e_sizes) || !PyArray_Check(r_sizes))
    {
        PyErr_SetString(PyExc_TypeError, "Input data must be numpy arrays");
        return NULL;
    }

    PyArrayObject *leaf_predictions_array = (PyArrayObject *)PyArray_FROM_OTF(leaf_predictions_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *e_array = (PyArrayObject *)PyArray_FROM_OTF(e_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *r_array = (PyArrayObject *)PyArray_FROM_OTF(r_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *e_sizes_array = (PyArrayObject *)PyArray_FROM_OTF(e_sizes, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *r_sizes_array = (PyArrayObject *)PyArray_FROM_OTF(r_sizes, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    std::string index_str = std::string(index_cptr);
    if (!leaf_predictions_array || !e_array || !r_array || !e_sizes_array || !r_sizes_array)
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(e_array);
        Py_XDECREF(r_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        PyErr_SetString(PyExc_TypeError, "Failed to convert input to numpy arrays");
        return NULL;
    }

    IndexType index;
    if (!parse_index_type(index_str, index))
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(e_array);
        Py_XDECREF(r_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + index_str).c_str());
        return NULL;
    }

    int64_t *e_data = (int64_t *)PyArray_DATA(e_array);
    int64_t *r_data = (int64_t *)PyArray_DATA(r_array);
    int64_t *e_sizes_data = (int64_t *)PyArray_DATA(e_sizes_array);
    int64_t *r_sizes_data = (int64_t *)PyArray_DATA(r_sizes_array);
    float *leaf_predictions = (float *)PyArray_DATA(leaf_predictions_array);

    if (PyArray_NDIM(leaf_predictions_array) != 1 ||
        PyArray_NDIM(e_array) != 2 ||
        PyArray_NDIM(r_array) != 2 ||
        PyArray_NDIM(e_sizes_array) != 1 ||
        PyArray_NDIM(r_sizes_array) != 1)
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(e_array);
        Py_XDECREF(r_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        PyErr_SetString(PyExc_ValueError, "Expected shapes: leaf_predictions=(n_nodes,), e=(n_nodes, n_features), r=(n_nodes, n_features), e_sizes=(n_nodes,), r_sizes=(n_nodes,)");
        return NULL;
    }

    int n_nodes = static_cast<int>(e_array->dimensions[0]);
    int n_nodes_r = static_cast<int>(r_array->dimensions[0]);
    int e_stride = static_cast<int>(e_array->dimensions[1]);
    int r_stride = static_cast<int>(r_array->dimensions[1]);

    if (n_nodes != n_nodes_r ||
        static_cast<int>(leaf_predictions_array->dimensions[0]) != n_nodes ||
        static_cast<int>(e_sizes_array->dimensions[0]) != n_nodes ||
        static_cast<int>(r_sizes_array->dimensions[0]) != n_nodes)
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(e_array);
        Py_XDECREF(r_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        PyErr_SetString(PyExc_ValueError, "Input arrays must share the same first dimension n_nodes");
        return NULL;
    }

    if (n_features <= 0 || e_stride < n_features || r_stride < n_features)
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(e_array);
        Py_XDECREF(r_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        PyErr_SetString(PyExc_ValueError, "Invalid n_features or incompatible e/r matrix width");
        return NULL;
    }

    // Initialize weight cache and buffers for processing E and R sets for each node.
    // The weight cache is initialized with a size based on the number of features, and we use fixed-size bit buffers for small sets of features (up to 64) and dynamic vectors for larger sets to efficiently store the indices of features in E and R for each node.
    inter_weights::WeightCache weight_cache = inter_weights::WeightCache((uint64_t)(2 * n_features + 1));
    algorithms::SparseInteractionMap sparse_result;

    uint64_t bit_buffer_E[64];
    uint64_t bit_buffer_R[64];
    std::vector<uint64_t> vector_buffer_E;
    std::vector<uint64_t> vector_buffer_R;
    uint64_t *e_buffer;
    uint64_t *r_buffer;
    int e_count, r_count;
    bool has_error = false;
    for (int i = 0; i < n_nodes; i++)
    {

        // Choose the appropriate buffer based on the size of E and R for the current node.
        e_count = static_cast<int>(e_sizes_data[i]);
        r_count = static_cast<int>(r_sizes_data[i]);
        if (e_count < 0 || r_count < 0 || e_count > e_stride || r_count > r_stride || e_count + r_count > n_features)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Invalid e/r sizes at node %d: e_count=%d, r_count=%d, e_stride=%d, r_stride=%d, n_features=%d",
                i,
                e_count,
                r_count,
                e_stride,
                r_stride,
                n_features);
            has_error = true;
            break;
        }
        if (e_count <= 64)
        {
            e_buffer = bit_buffer_E;
        }
        else
        {
            vector_buffer_E.resize(e_count);
            e_buffer = vector_buffer_E.data();
        }
        if (r_count <= 64)
        {
            r_buffer = bit_buffer_R;
        }
        else
        {
            vector_buffer_R.resize(r_count);
            r_buffer = vector_buffer_R.data();
        }

        for (int j = 0; j < e_count; j++)
        {
            int64_t feature_index = e_data[i * e_stride + j];
            if (feature_index < 0 || feature_index >= n_features)
            {
                PyErr_Format(
                    PyExc_ValueError,
                    "Invalid feature index in E at node %d, pos %d: %lld (expected in [0, %d))",
                    i,
                    j,
                    static_cast<long long>(feature_index),
                    n_features);
                has_error = true;
                break;
            }
            e_buffer[j] = static_cast<uint64_t>(feature_index);
        }
        if (has_error)
        {
            break;
        }
        for (int j = 0; j < r_count; j++)
        {
            int64_t feature_index = r_data[i * r_stride + j];
            if (feature_index < 0 || feature_index >= n_features)
            {
                PyErr_Format(
                    PyExc_ValueError,
                    "Invalid feature index in R at node %d, pos %d: %lld (expected in [0, %d))",
                    i,
                    j,
                    static_cast<long long>(feature_index),
                    n_features);
                has_error = true;
                break;
            }
            r_buffer[j] = static_cast<uint64_t>(feature_index);
        }
        if (has_error)
        {
            break;
        }
        // Get leaf value and compute contributions for all subsets of E and R up to the specified max_order.
        // We iterate over all possible subset sizes s from 1 to max_order.
        // For each subset size s, we determine how many features in the subset come from E (s_cap_e) and how many come from R (s_cap_r).
        // We then compute the weight for that combination of features using the weight cache and call the recursive enumeration functions to generate all subsets of E and R with the specified number of features, updating the interactions map with the contributions.
        float leaf_value = leaf_predictions[i];
        BitSet subset(n_features);
        for (int s = 1; s <= max_order; ++s)
        {
            int min_from_e = std::max(0, s - static_cast<int>(r_count));
            int max_from_e = std::min(s, static_cast<int>(e_count));
            for (int s_cap_e = min_from_e; s_cap_e <= max_from_e; ++s_cap_e)
            {
                int s_cap_r = s - s_cap_e;
                const double weight = weight_cache.get_weight(n_features, e_count, r_count, s_cap_e, s_cap_r, s, index, max_order);
                if (weight == 0.0)
                {
                    continue;
                }
                const double contribution = static_cast<double>(leaf_value) * weight;
                if (std::abs(contribution) < 1e-12) // Skip negligible contributions to save time on enumeration and map updates.
                {
                    continue;
                }
                // We now update all the interactions corresponding to subsets of E and R with s_cap_e features from E and s_cap_r features from R by calling the enumerate_e_subsets function, which will recursively generate all such subsets and update the interactions map with the computed contribution for each subset.
                algorithms::enumerate_e_subsets(
                    e_buffer,
                    static_cast<int>(e_count),
                    r_buffer,
                    static_cast<int>(r_count),
                    0,
                    s_cap_e,
                    s_cap_r,
                    subset,
                    contribution,
                    sparse_result);
            }
        }
    }
    if (has_error)
    {
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(e_array);
        Py_XDECREF(r_array);
        Py_XDECREF(e_sizes_array);
        Py_XDECREF(r_sizes_array);
        return NULL;
    }
    PyObject *output = sparse_map_to_pydict(sparse_result);

    Py_XDECREF(leaf_predictions_array);
    Py_XDECREF(e_array);
    Py_XDECREF(r_array);
    Py_XDECREF(e_sizes_array);
    Py_XDECREF(r_sizes_array);
    return output;
}
