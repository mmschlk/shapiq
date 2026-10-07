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
static PyObject *compute_interactions_cohort(PyObject *self, PyObject *args);
static PyObject *predict_ensemble_sum(PyObject *self, PyObject *args);
static PyObject *preprocess_subset_tables(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"compute_interactions_batched_sparse", compute_interactions_batched_sparse, METH_VARARGS, "Compute sparse feature interactions in batches using the interventional algorithm."},
    {"compute_interactions_cohort", compute_interactions_cohort, METH_VARARGS, "Cohort DFS sharing one tree walk across all reference samples, accumulating every order over the structural subset layout into `out`."},
    {"predict_ensemble_sum", predict_ensemble_sum, METH_VARARGS, "Route every row of X through every tree and return the per-row sum of leaf predictions."},
    {"preprocess_subset_tables", preprocess_subset_tables, METH_VARARGS, "From the ensemble block's features, children_left, children_right and tree_offsets: per order >= 2 the sorted feature subsets co-occurring on some root-to-leaf path and their hash index, ((keys, starts, counts), (slot_keys, slot_rows, block_starts, block_shifts)), or None when the keys overflow."},
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

// Converts the flat map to {sorted feature tuple: value / n_reference_samples}. Each key's
// digits in base n_features + 1 are (feature + 1), last feature lowest.
static PyObject *flat_map_to_pydict(const algorithms::FlatSubsetMap &result, int n_features,
                                    int n_reference_samples)
{
    PyObject *output = PyDict_New();
    if (!output)
    {
        return NULL;
    }
    const int64_t base = static_cast<int64_t>(n_features) + 1;
    std::vector<PyObject *> feature_ids(static_cast<size_t>(n_features), nullptr);
    bool failed = false;
    int64_t digits[64];
    result.for_each(
        [&](int64_t key, double value)
        {
            if (failed)
                return;
            int size = 0;
            for (int64_t rest = key; rest > 0; rest /= base)
                digits[size++] = rest % base - 1;
            PyObject *tuple = PyTuple_New(size);
            if (!tuple)
            {
                failed = true;
                return;
            }
            for (int i = 0; i < size; ++i)
            {
                const int64_t feature = digits[size - 1 - i];
                PyObject *&id = feature_ids[static_cast<size_t>(feature)];
                if (!id && !(id = PyLong_FromLongLong(feature)))
                {
                    Py_DECREF(tuple);
                    failed = true;
                    return;
                }
                Py_INCREF(id);
                PyTuple_SET_ITEM(tuple, i, id);
            }
            PyObject *item = PyFloat_FromDouble(value / static_cast<double>(n_reference_samples));
            if (!item || PyDict_SetItem(output, tuple, item) < 0)
                failed = true;
            Py_XDECREF(item);
            Py_DECREF(tuple);
        });
    for (PyObject *id : feature_ids)
    {
        Py_XDECREF(id);
    }
    if (failed)
    {
        Py_DECREF(output);
        return NULL;
    }
    return output;
}

// === ensemble node block ===
// All trees' node arrays back to back, node ids tree-relative: tree t occupies
// [tree_offsets[t], tree_offsets[t + 1]) and its categorical split values
// [cat_offsets[t], cat_offsets[t + 1]) of cat_values (cat_start is relative to that segment).
// Built once by InterventionalTreeSHAPIQ._preprocess_trees; every kernel takes it as its
// first eleven arguments and views each tree through offset pointers.
struct EnsembleBlock
{
    std::vector<PyArrayObject *> owned;
    std::vector<Tree> trees;
    int64_t n_nodes = 0;
    const int64_t *features = nullptr;
    const int64_t *children_left = nullptr;
    const int64_t *children_right = nullptr;
    const int64_t *tree_offsets = nullptr;

    ~EnsembleBlock()
    {
        for (PyArrayObject *arr : owned)
            Py_XDECREF(arr);
    }

    PyArrayObject *take(PyObject *obj, int np_type)
    {
        PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(obj, np_type, NPY_ARRAY_IN_ARRAY);
        if (arr)
            owned.push_back(arr);
        return arr;
    }

    // Parses and validates the block; n_features < 0 skips the split-feature range check.
    // Sets a Python error and returns false on failure.
    bool parse(PyObject *values_obj, PyObject *thresholds_obj, PyObject *features_obj,
               PyObject *children_left_obj, PyObject *children_right_obj, PyObject *children_missing_obj,
               PyObject *cat_values_obj, PyObject *cat_start_obj, PyObject *cat_size_obj,
               PyObject *tree_offsets_obj, PyObject *cat_offsets_obj,
               const char *decision_type, int n_features)
    {
        PyArrayObject *values = take(values_obj, NPY_FLOAT64);
        PyArrayObject *thresholds = take(thresholds_obj, NPY_FLOAT64);
        PyArrayObject *feats = take(features_obj, NPY_INT64);
        PyArrayObject *cl = take(children_left_obj, NPY_INT64);
        PyArrayObject *cr = take(children_right_obj, NPY_INT64);
        PyArrayObject *cm = take(children_missing_obj, NPY_BOOL);
        PyArrayObject *cat_values = take(cat_values_obj, NPY_INT64);
        PyArrayObject *cat_start = take(cat_start_obj, NPY_INT64);
        PyArrayObject *cat_size = take(cat_size_obj, NPY_INT64);
        PyArrayObject *toff = take(tree_offsets_obj, NPY_INT64);
        PyArrayObject *coff = take(cat_offsets_obj, NPY_INT64);
        PyArrayObject *all[] = {values, thresholds, feats, cl, cr, cm, cat_values, cat_start, cat_size, toff, coff};
        for (PyArrayObject *arr : all)
        {
            if (!arr || PyArray_NDIM(arr) != 1)
            {
                PyErr_SetString(PyExc_TypeError, "ensemble block arrays must be 1-dimensional numpy arrays");
                return false;
            }
        }
        n_nodes = (int64_t)PyArray_DIM(values, 0);
        const int64_t n_cat = (int64_t)PyArray_DIM(cat_values, 0);
        const int64_t n_trees = (int64_t)PyArray_DIM(toff, 0) - 1;
        for (PyArrayObject *arr : {thresholds, feats, cl, cr, cm, cat_start, cat_size})
        {
            if ((int64_t)PyArray_DIM(arr, 0) != n_nodes)
            {
                PyErr_SetString(PyExc_ValueError, "ensemble block node arrays must all have one entry per node");
                return false;
            }
        }
        const int64_t *t_off = (const int64_t *)PyArray_DATA(toff);
        const int64_t *c_off = (const int64_t *)PyArray_DATA(coff);
        if (n_trees < 1 || (int64_t)PyArray_DIM(coff, 0) != n_trees + 1 || t_off[0] != 0 || c_off[0] != 0 ||
            t_off[n_trees] != n_nodes || c_off[n_trees] != n_cat)
        {
            PyErr_SetString(PyExc_ValueError, "ensemble block offsets must start at 0, be one longer than the number of trees, and end at the array lengths");
            return false;
        }
        features = (const int64_t *)PyArray_DATA(feats);
        children_left = (const int64_t *)PyArray_DATA(cl);
        children_right = (const int64_t *)PyArray_DATA(cr);
        tree_offsets = t_off;
        const int64_t *cstart = (const int64_t *)PyArray_DATA(cat_start);
        const int64_t *csize = (const int64_t *)PyArray_DATA(cat_size);
        trees.reserve(static_cast<size_t>(n_trees));
        for (int64_t t = 0; t < n_trees; ++t)
        {
            const int64_t off = t_off[t], n_t = t_off[t + 1] - off, cat_n = c_off[t + 1] - c_off[t];
            if (n_t < 1 || cat_n < 0)
            {
                PyErr_SetString(PyExc_ValueError, "ensemble block offsets must be increasing (every tree has at least one node)");
                return false;
            }
            for (int64_t i = off; i < off + n_t; ++i)
            {
                const bool leaf = children_left[i] < 0;
                if ((children_right[i] < 0) != leaf || (!leaf && (children_left[i] >= n_t || children_right[i] >= n_t)) ||
                    (!leaf && n_features >= 0 && (features[i] < 0 || features[i] >= n_features)) ||
                    csize[i] < 0 || (csize[i] > 0 && (cstart[i] < 0 || cstart[i] + csize[i] > cat_n)))
                {
                    PyErr_SetString(PyExc_ValueError, "ensemble block has a child, feature or categorical index out of range");
                    return false;
                }
            }
            trees.push_back(Tree(
                (double *)PyArray_DATA(values) + off,
                (double *)PyArray_DATA(thresholds) + off,
                (int64_t *)features + off,
                (int64_t *)children_left + off,
                (int64_t *)children_right + off,
                (bool *)PyArray_DATA(cm) + off,
                std::string(decision_type),
                (const int64_t *)PyArray_DATA(cat_values) + c_off[t],
                cstart + off,
                csize + off));
        }
        return true;
    }
};

static PyObject *compute_interactions_batched_sparse(PyObject *self, PyObject *args)
{
    /**
     * Sparse per-explanation kernel: one DFS per (tree, reference row) accumulating every
     * order into a per-thread subset map (FlatSubsetMap, or the BitSet map when the keys
     * overflow), returned as {feature tuple: value}. Arguments: the ensemble block (see
     * EnsembleBlock), reference_data (n_ref x n_features float64), explain_data (float64),
     * decision_type, index, max_order, verbose, and an optional custom weight table.
     */
    PyObject *values_obj, *thresholds_obj, *features_obj, *children_left_obj, *children_right_obj,
        *children_missing_obj, *cat_values_obj, *cat_start_obj, *cat_size_obj, *tree_offsets_obj,
        *cat_offsets_obj;
    PyObject *reference_data_obj;
    PyObject *explain_data_obj;
    const char *decision_type_cptr;
    const char *index_cptr;
    int max_order;
    int verbose;
    PyObject *weight_table_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOssii|O", &values_obj, &thresholds_obj, &features_obj, &children_left_obj, &children_right_obj, &children_missing_obj, &cat_values_obj, &cat_start_obj, &cat_size_obj, &tree_offsets_obj, &cat_offsets_obj, &reference_data_obj, &explain_data_obj, &decision_type_cptr, &index_cptr, &max_order, &verbose, &weight_table_obj))
        return NULL;
    if (max_order < 1)
    {
        PyErr_SetString(PyExc_ValueError, "max_order must be >= 1");
        return NULL;
    }
    PyArrayObject *reference_data_array = (PyArrayObject *)PyArray_FROM_OTF(reference_data_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *explain_data_array = (PyArrayObject *)PyArray_FROM_OTF(explain_data_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *weight_table_array = (weight_table_obj != Py_None) ? (PyArrayObject *)PyArray_FROM_OTF(weight_table_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY) : nullptr;
    auto release = [&]()
    {
        Py_XDECREF(reference_data_array);
        Py_XDECREF(explain_data_array);
        Py_XDECREF(weight_table_array);
    };
    if (!reference_data_array || !explain_data_array || PyArray_NDIM(reference_data_array) != 2 ||
        (weight_table_obj != Py_None && !weight_table_array))
    {
        release();
        PyErr_SetString(PyExc_TypeError, "reference_data must be a 2-D float64 numpy array, explain_data a float64 numpy array, weight_table a float64 numpy array or None");
        return NULL;
    }
    double *reference_data = (double *)PyArray_DATA(reference_data_array);
    double *explain_data = (double *)PyArray_DATA(explain_data_array);
    int n_reference_samples = static_cast<int>(PyArray_DIM(reference_data_array, 0));
    int n_features = static_cast<int>(PyArray_DIM(reference_data_array, 1));
    EnsembleBlock block;
    if (!block.parse(values_obj, thresholds_obj, features_obj, children_left_obj, children_right_obj, children_missing_obj, cat_values_obj, cat_start_obj, cat_size_obj, tree_offsets_obj, cat_offsets_obj, decision_type_cptr, n_features))
    {
        release();
        return NULL;
    }
    std::vector<Tree> &trees = block.trees;
    const int num_trees = static_cast<int>(trees.size());
    IndexType index_type;
    if (!parse_index_type(std::string(index_cptr), index_type))
    {
        release();
        PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + std::string(index_cptr)).c_str());
        return NULL;
    }
    const double *custom_table = nullptr;
    int64_t custom_N = 0, custom_K = 0;
    if (weight_table_array)
    {
        custom_table = (const double *)PyArray_DATA(weight_table_array);
        custom_N = (int64_t)n_features + 1;
        custom_K = (int64_t)max_order + 1;
    }

    PyObject *output = NULL;
    if (algorithms::flat_subset_keys_fit(n_features, max_order))
    {
        // one flat map per thread, filled with exactly the subsets this explanation's E/R sets
        // produce, then merged into the largest one (deterministic for a fixed schedule)
        std::vector<algorithms::FlatSubsetMap> thread_maps(static_cast<size_t>(omp_get_max_threads()));
        algorithms::FlatSubsetMap *merged = nullptr;
        Py_BEGIN_ALLOW_THREADS
#pragma omp parallel
        {
            // filled on this thread's own stack: maps side by side in `thread_maps` would share
            // cache lines, and every insert's size update would bounce them between cores
            algorithms::FlatSubsetMap local;
            std::vector<StackFrame> stack;  // traversal scratch, reused for every (tree, reference)
            inter_weights::WeightCache weight_cache = (custom_table != nullptr)
                                                          ? inter_weights::WeightCache((uint64_t)(2 * n_features), custom_table, custom_N, custom_K)
                                                          : inter_weights::WeightCache((uint64_t)(2 * n_features));

#pragma omp for nowait
            for (int t = 0; t < num_trees; t++)
            {
                for (int i = 0; i < n_reference_samples; i++)
                {
                    algorithms::compute_interactions_flat(
                        trees[t],
                        local,
                        weight_cache,
                        reference_data + i * n_features,
                        explain_data,
                        n_features,
                        index_type,
                        max_order,
                        stack);
                }
            }
            thread_maps[static_cast<size_t>(omp_get_thread_num())] = std::move(local);
        }
        merged = &*std::max_element(thread_maps.begin(), thread_maps.end(),
                                    [](const algorithms::FlatSubsetMap &a, const algorithms::FlatSubsetMap &b)
                                    { return a.size() < b.size(); });
        for (const auto &map : thread_maps)
        {
            if (&map != merged)
                merged->merge_from(map);
        }
        Py_END_ALLOW_THREADS
        output = flat_map_to_pydict(*merged, n_features, n_reference_samples);
    }
    else
    {
        // (n_features + 1)^max_order overflows an int64: keep the BitSet-keyed map
        algorithms::SparseInteractionMap sparse_result;
        Py_BEGIN_ALLOW_THREADS
#pragma omp parallel
        {
            algorithms::SparseInteractionMap local_sparse_result;
            std::vector<StackFrame> stack;  // traversal scratch, reused for every (tree, reference)
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
                        verbose,
                        stack);
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

        output = sparse_map_to_pydict(sparse_result);
    }

    release();
    return output;
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

// `leaf_update(leaf_feats, leaf_fie, n_leaf_feats, e, r, scaled_val)` receives each leaf's
// E features (fie = 1) then R features (fie = 0), both ascending, and the leaf value scaled
// by the cohort size.
template <typename LeafUpdate>
static void cohort_walk(
    Tree &tree,
    const double *__restrict__ reference_data,
    const double *__restrict__ explain_data,
    int n_ref,
    int n_features,
    double inv_scaling,
    std::vector<int> &ref_order,       // scratch, size n_ref
    std::vector<int32_t> &leaf_feats,  // scratch, size >= max depth
    std::vector<int32_t> &leaf_fie,    // scratch, size >= max depth
    std::vector<CohortFrame> &stack,   // scratch
    LeafUpdate &&leaf_update)
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
            leaf_update(leaf_feats.data(), leaf_fie.data(), n_leaf_feats, e, r, scaled_val);
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

// === structural layout ===
// Output over the subsets that co-occur on some path (preprocess_subset_tables): a dense
// order-1 block [0, n_features), then per order >= 2 the rows of its table at
// order_offsets[order]. A subset's row is found through the flat hash index
// (shapiq.tree.subset_index.build_subset_index); keys are the sorted features as base-
// n_features digits.
struct StructuralLayout
{
    int n_features;
    int max_order;
    const int64_t *counts;        // per order: rows in its table
    const int64_t *order_offsets; // per order: start of its block in the output
    const int64_t *slot_keys;     // hash index: slot keys, -1 = empty
    const int32_t *slot_rows;     // hash index: slot row ids
    const int64_t *block_starts;  // per order: first slot of its block
    const int64_t *block_shifts;  // per order: 64 - log2(block size)

    // row of a sorted order-s subset with the given key, or -1 when it is not in the table
    int64_t row(int64_t key, int s) const
    {
        const int64_t block = block_starts[s];
        const int shift = static_cast<int>(block_shifts[s]);
        const uint64_t mask = (uint64_t(1) << (64 - shift)) - 1;
        uint64_t slot = (static_cast<uint64_t>(key) * algorithms::kIndexHashMultiplier) >> shift;
        for (uint64_t probe = 0; probe <= mask; ++probe)
        {
            const int64_t stored = slot_keys[block + slot];
            if (stored == key)
            {
                const int64_t r = slot_rows[block + slot];
                return (r >= 0 && r < counts[s]) ? r : -1;
            }
            if (stored < 0)
                return -1;
            slot = (slot + 1) & mask;
        }
        return -1;
    }
};

// Weights for every (|E|, |R|, s_cap_e, s_cap_r) a leaf can produce, computed once per call:
// w[((e * stride + r) * k1 + s_cap_e) * k1 + s_cap_r], k1 = max_order + 1.
static std::vector<double> structural_weight_table(
    inter_weights::WeightCache &weight_cache, IndexType index_type, int n_features,
    int max_order, int stride)
{
    const int k1 = max_order + 1;
    std::vector<double> w(static_cast<size_t>(stride) * stride * k1 * k1, 0.0);
    for (int e = 0; e < stride; ++e)
        for (int r = 0; r < stride; ++r)
            for (int sce = 0; sce <= std::min(e, max_order); ++sce)
                for (int scr = 0; scr <= std::min(r, max_order - sce); ++scr)
                {
                    const int s = sce + scr;
                    if (s == 0)
                        continue;
                    w[((static_cast<size_t>(e) * stride + r) * k1 + sce) * k1 + scr] =
                        weight_cache.get_weight(n_features, e, r, sce, scr, s, index_type, max_order);
                }
    return w;
}

// One leaf: every subset of E u R up to max_order, weighted by (|E|, |R|, s_cap_e, s_cap_r),
// added at its row of the structural layout. Features are sorted first so that each
// enumerated subset is sorted and its key accumulates digit by digit.
struct StructuralLeafUpdate
{
    const StructuralLayout &layout;
    const double *weights;  // structural_weight_table
    int stride;
    double *local;          // this thread's output row
    bool missing = false;   // a subset was not in the tables (tables and walk disagree)
    int32_t feats[64];
    int32_t fie[64];

    void operator()(const int32_t *leaf_feats, const int32_t *leaf_fie, int n, int e, int r, double scaled_val)
    {
        if (n > 64)
        {
            missing = true;  // deeper paths than the layout supports
            return;
        }
        for (int i = 0; i < n; ++i)  // insertion sort by feature, carrying the E/R flag
        {
            int32_t f = leaf_feats[i], g = leaf_fie[i];
            int j = i;
            while (j > 0 && feats[j - 1] > f)
            {
                feats[j] = feats[j - 1];
                fie[j] = fie[j - 1];
                --j;
            }
            feats[j] = f;
            fie[j] = g;
        }
        const int k1 = layout.max_order + 1;
        const double *w_er = weights + (static_cast<size_t>(e) * stride + r) * k1 * k1;
        if (layout.max_order <= 3)
        {
            // the common orders unrolled: no recursion, keys built incrementally
            const int64_t F = layout.n_features;
            const int K = layout.max_order;
            for (int i = 0; i < n; ++i)
            {
                const int32_t fi = feats[i];
                const int ei = fie[i];
                add(1, fi, scaled_val * w_er[ei * k1 + (1 - ei)], fi);
                if (K < 2)
                    continue;
                const int64_t key_i = static_cast<int64_t>(fi) * F;
                for (int j = i + 1; j < n; ++j)
                {
                    const int eij = ei + fie[j];
                    const int64_t key_ij = key_i + feats[j];
                    add(2, key_ij, scaled_val * w_er[eij * k1 + (2 - eij)], -1);
                    if (K < 3)
                        continue;
                    const int64_t key_ij_f = key_ij * F;
                    for (int k = j + 1; k < n; ++k)
                    {
                        const int eijk = eij + fie[k];
                        add(3, key_ij_f + feats[k], scaled_val * w_er[eijk * k1 + (3 - eijk)], -1);
                    }
                }
            }
            return;
        }
        enumerate(0, 0, 0, 0, n, w_er, k1, scaled_val);
    }

    inline void add(int s, int64_t key, double value, int32_t feature)
    {
        if (value == 0.0)
            return;
        if (s == 1)
        {
            local[feature] += value;
            return;
        }
        const int64_t row = layout.row(key, s);
        if (row >= 0)
            local[layout.order_offsets[s] + row] += value;
        else
            missing = true;
    }

    // chosen so far: `level` features, `sce` of them from E, key = their digits
    void enumerate(int start, int level, int sce, int64_t key, int n, const double *w_er, int k1, double scaled_val)
    {
        for (int i = start; i < n; ++i)
        {
            const int s = level + 1;
            const int sce_new = sce + fie[i];
            const int64_t key_new = key * layout.n_features + feats[i];
            const double w = w_er[sce_new * k1 + (s - sce_new)];
            if (w != 0.0)
            {
                if (s == 1)
                    local[feats[i]] += scaled_val * w;
                else
                {
                    const int64_t row = layout.row(key_new, s);
                    if (row >= 0)
                        local[layout.order_offsets[s] + row] += scaled_val * w;
                    else
                        missing = true;
                }
            }
            if (s < layout.max_order)
                enumerate(i + 1, s, sce_new, key_new, n, w_er, k1, scaled_val);
        }
    }
};

// Structural result -> {sorted feature tuple: value}, skipping zeros: the order-1 block by
// feature id, then each table row's features (subset_keys, the layout of
// preprocess_subset_tables).
static PyObject *structural_result_to_dict(const double *out, int n_features, int max_order,
                                           const int32_t *subset_keys, const int64_t *subset_starts,
                                           const int64_t *counts, const int64_t *order_offsets)
{
    PyObject *output = PyDict_New();
    if (!output)
        return NULL;
    std::vector<PyObject *> ids(static_cast<size_t>(n_features), nullptr);
    auto feature_id = [&](int32_t f) -> PyObject *
    {
        PyObject *&id = ids[static_cast<size_t>(f)];
        if (!id)
            id = PyLong_FromLong(f);
        if (id)
            Py_INCREF(id);
        return id;  // new reference, or NULL
    };
    bool failed = false;
    auto put = [&](PyObject *key, double value)
    {
        PyObject *item = key ? PyFloat_FromDouble(value) : NULL;
        if (!item || PyDict_SetItem(output, key, item) < 0)
            failed = true;
        Py_XDECREF(item);
        Py_XDECREF(key);
    };
    for (int f = 0; f < n_features && !failed; ++f)
    {
        if (out[f] == 0.0)
            continue;
        PyObject *key = PyTuple_New(1);
        if (key)
            PyTuple_SET_ITEM(key, 0, feature_id(static_cast<int32_t>(f)));
        put(key, out[f]);
    }
    for (int order = 2; order <= max_order && !failed; ++order)
    {
        const int32_t *row = subset_keys + subset_starts[order];
        const double *values = out + order_offsets[order];
        for (int64_t r = 0; r < counts[order] && !failed; ++r, row += order)
        {
            if (values[r] == 0.0)
                continue;
            PyObject *key = PyTuple_New(order);
            for (int i = 0; key && i < order; ++i)
                PyTuple_SET_ITEM(key, i, feature_id(row[i]));
            put(key, values[r]);
        }
    }
    for (PyObject *id : ids)
        Py_XDECREF(id);
    if (failed || PyErr_Occurred())
    {
        Py_DECREF(output);
        return NULL;
    }
    return output;
}

// Cohort walk over every tree writing into the structural layout: one output row per
// thread in a contiguous (threads x rows) block, merged afterwards by a parallel
// vectorised column sum into `out`. Returns None, or raises when a subset enumerated at a
// leaf is missing from the tables (the tables and the walk disagree).
template <typename Cleanup>
static PyObject *compute_interactions_structural(
    std::vector<Tree> &trees, const double *reference_data, const double *explain_data,
    int n_ref, int n_features, IndexType index_type, int max_order,
    const double *custom_table, int64_t custom_N, int64_t custom_K,
    PyObject *subset_keys_obj, PyObject *subset_starts_obj, PyObject *counts_obj,
    PyObject *slot_keys_obj, PyObject *slot_rows_obj, PyObject *block_starts_obj,
    PyObject *block_shifts_obj, PyObject *out_obj, Cleanup &cleanup_arrays)
{
    PyArrayObject *subset_keys_arr = (PyArrayObject *)PyArray_FROM_OTF(subset_keys_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *subset_starts_arr = (PyArrayObject *)PyArray_FROM_OTF(subset_starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *counts_arr = (PyArrayObject *)PyArray_FROM_OTF(counts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *slot_keys_arr = (PyArrayObject *)PyArray_FROM_OTF(slot_keys_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *slot_rows_arr = (PyArrayObject *)PyArray_FROM_OTF(slot_rows_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *block_starts_arr = (PyArrayObject *)PyArray_FROM_OTF(block_starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *block_shifts_arr = (PyArrayObject *)PyArray_FROM_OTF(block_shifts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_FROM_OTF(out_obj, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY2);
    auto release = [&]()
    {
        Py_XDECREF(subset_keys_arr);
        Py_XDECREF(subset_starts_arr);
        Py_XDECREF(counts_arr);
        Py_XDECREF(slot_keys_arr);
        Py_XDECREF(slot_rows_arr);
        Py_XDECREF(block_starts_arr);
        Py_XDECREF(block_shifts_arr);
        if (out_arr)
            PyArray_ResolveWritebackIfCopy(out_arr);
        Py_XDECREF(out_arr);
        cleanup_arrays();
    };
    if (!subset_keys_arr || !subset_starts_arr || !counts_arr || !slot_keys_arr || !slot_rows_arr || !block_starts_arr || !block_shifts_arr || !out_arr)
    {
        release();
        PyErr_SetString(PyExc_TypeError, "structural layout arrays must be numpy arrays (int64 counts/index, int32 rows, float64 out)");
        return NULL;
    }
    const char *arg_error = NULL;
    const int64_t n_orders = max_order + 1;
    if (PyArray_NDIM(subset_keys_arr) != 1 || PyArray_NDIM(subset_starts_arr) != 1 ||
        PyArray_NDIM(counts_arr) != 1 || PyArray_NDIM(slot_keys_arr) != 1 || PyArray_NDIM(slot_rows_arr) != 1 ||
        PyArray_NDIM(block_starts_arr) != 1 || PyArray_NDIM(block_shifts_arr) != 1 || PyArray_NDIM(out_arr) != 1)
        arg_error = "structural layout arrays must be 1-dimensional.";
    else if (PyArray_DIM(counts_arr, 0) != n_orders || PyArray_DIM(subset_starts_arr, 0) != n_orders ||
             PyArray_DIM(block_starts_arr, 0) != n_orders ||
             PyArray_DIM(block_shifts_arr, 0) != n_orders || PyArray_DIM(slot_rows_arr, 0) != PyArray_DIM(slot_keys_arr, 0))
        arg_error = "structural layout arrays have inconsistent lengths.";
    const int64_t *counts = arg_error ? NULL : (const int64_t *)PyArray_DATA(counts_arr);
    std::vector<int64_t> order_offsets(static_cast<size_t>(n_orders), 0);
    int64_t n_rows = n_features;
    if (!arg_error)
    {
        const int64_t n_slots = (int64_t)PyArray_DIM(slot_keys_arr, 0);
        const int64_t *bstarts = (const int64_t *)PyArray_DATA(block_starts_arr);
        const int64_t *bshifts = (const int64_t *)PyArray_DATA(block_shifts_arr);
        int64_t cap = 1;
        for (int i = 0; !arg_error && i < max_order; ++i)
        {
            if (cap > INT64_MAX / n_features)
                arg_error = "the subset key encoding overflows for this order.";
            else
                cap *= n_features;
        }
        const int64_t *sstarts = (const int64_t *)PyArray_DATA(subset_starts_arr);
        const int64_t n_keys = (int64_t)PyArray_DIM(subset_keys_arr, 0);
        for (int order = 2; !arg_error && order <= max_order; ++order)
        {
            order_offsets[order] = n_rows;
            if (counts[order] < 0 || sstarts[order] < 0 || sstarts[order] + counts[order] * order > n_keys)
                arg_error = "subset tables are inconsistent with the subset_keys length.";
            else if (bshifts[order] < 2 || bshifts[order] > 63 || bstarts[order] < 0 ||
                     bstarts[order] > n_slots - (int64_t(1) << (64 - bshifts[order])))
                arg_error = "subset index blocks lie outside the index arrays.";
            n_rows += counts[order];
        }
        if (!arg_error && PyArray_DIM(out_arr, 0) != n_rows)
            arg_error = "out must have n_features + (number of table rows) entries.";
    }
    if (arg_error)
    {
        release();
        PyErr_SetString(PyExc_ValueError, arg_error);
        return NULL;
    }

    StructuralLayout layout;
    layout.n_features = n_features;
    layout.max_order = max_order;
    layout.counts = counts;
    layout.order_offsets = order_offsets.data();
    layout.slot_keys = (const int64_t *)PyArray_DATA(slot_keys_arr);
    layout.slot_rows = (const int32_t *)PyArray_DATA(slot_rows_arr);
    layout.block_starts = (const int64_t *)PyArray_DATA(block_starts_arr);
    layout.block_shifts = (const int64_t *)PyArray_DATA(block_shifts_arr);
    double *out = (double *)PyArray_DATA(out_arr);
    std::fill(out, out + n_rows, 0.0);
    const double inv_scaling = (n_ref > 0) ? 1.0 / (double)n_ref : 0.0;
    bool missing = false;

    Py_BEGIN_ALLOW_THREADS
    int max_depth = 0;
    for (size_t t = 0; t < trees.size(); t++)
        max_depth = std::max(max_depth, cohort_max_depth(trees[t]));
    max_depth = std::min(max_depth, n_features);
    const int stride = max_depth + 1;
    inter_weights::WeightCache table_cache = (custom_table != nullptr)
                                                 ? inter_weights::WeightCache((uint64_t)(3 * n_features), custom_table, custom_N, custom_K)
                                                 : inter_weights::WeightCache((uint64_t)(3 * n_features));
    const std::vector<double> weights = structural_weight_table(table_cache, index_type, n_features, max_order, stride);

    const int n_threads = omp_get_max_threads();
    std::vector<double> block(static_cast<size_t>(n_threads) * n_rows, 0.0);
    std::vector<char> thread_missing(static_cast<size_t>(n_threads), 0);
    const Py_ssize_t num_trees = (Py_ssize_t)trees.size();
    // Small blocks are merged serially as each thread finishes (no barrier: a barrier on
    // many threads costs more than summing a few thousand doubles); large blocks wait for
    // all rows and then sum in parallel, one slice of the rows per thread (vectorised).
    const bool parallel_merge = block.size() > static_cast<size_t>(1) << 16;
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        StructuralLeafUpdate leaf{layout, weights.data(), stride, block.data() + static_cast<size_t>(tid) * n_rows};
        std::vector<int> ref_order(n_ref);
        std::vector<int32_t> leaf_feats(max_depth + 1);
        std::vector<int32_t> leaf_fie(max_depth + 1);
        std::vector<CohortFrame> stack;
        stack.reserve(256);

        if (parallel_merge)
        {
#pragma omp for schedule(dynamic, 1)
            for (Py_ssize_t t = 0; t < num_trees; t++)
            {
                cohort_walk(trees[t], reference_data, explain_data, n_ref, n_features, inv_scaling,
                            ref_order, leaf_feats, leaf_fie, stack, leaf);
            }
            // the loop's implicit barrier: every row is complete before the merge; the
            // region's end joins after it (nowait avoids a second barrier)
#pragma omp for schedule(static) nowait
            for (int64_t k = 0; k < n_rows; k++)
            {
                double acc = 0.0;
                for (int th = 0; th < n_threads; th++)
                    acc += block[static_cast<size_t>(th) * n_rows + k];
                out[k] = acc;
            }
        }
        else
        {
#pragma omp for schedule(dynamic, 1) nowait
            for (Py_ssize_t t = 0; t < num_trees; t++)
            {
                cohort_walk(trees[t], reference_data, explain_data, n_ref, n_features, inv_scaling,
                            ref_order, leaf_feats, leaf_fie, stack, leaf);
            }
#pragma omp critical
            {
                for (int64_t k = 0; k < n_rows; k++)
                    out[k] += leaf.local[k];
            }
        }
        thread_missing[static_cast<size_t>(tid)] = leaf.missing ? 1 : 0;
    }
    for (char flag : thread_missing)
        missing = missing || (flag != 0);
    Py_END_ALLOW_THREADS

    PyObject *output = NULL;
    if (missing)
        PyErr_SetString(PyExc_RuntimeError, "a subset reached by the walk is missing from the subset tables (preprocess_subset_tables and the cohort walk disagree), or a path holds more than 64 features.");
    else
        output = structural_result_to_dict(out, n_features, max_order,
                                           (const int32_t *)PyArray_DATA(subset_keys_arr),
                                           (const int64_t *)PyArray_DATA(subset_starts_arr),
                                           counts, order_offsets.data());
    release();
    return output;
}

static PyObject *compute_interactions_cohort(PyObject *self, PyObject *args)
{
    /**
     * Structural kernel: one cohort DFS per tree shared across the reference rows,
     * accumulating every order over the structural subset layout (preprocess_subset_tables),
     * returned as {feature tuple: value}. Arguments: the ensemble block (see EnsembleBlock),
     * reference_data, explain_point, decision_type, index, max_order, verbose, the custom
     * weight table or None, then the subset tables (keys, starts, counts), the hash index
     * (slot_keys, slot_rows, block_starts, block_shifts) and a float64 scratch array `out`
     * of n_features + sum(counts) entries.
     */
    PyObject *values_obj, *thresholds_obj, *features_obj, *children_left_obj, *children_right_obj,
        *children_missing_obj, *cat_values_obj, *cat_start_obj, *cat_size_obj, *tree_offsets_obj,
        *cat_offsets_obj;
    PyObject *reference_data_obj;
    PyObject *explain_point_obj;
    const char *decision_type_cptr;
    const char *index_cptr;
    int max_order;
    int verbose;
    PyObject *weight_table_obj;
    PyObject *subset_keys_obj, *subset_starts_obj, *counts_obj, *slot_keys_obj, *slot_rows_obj,
        *block_starts_obj, *block_shifts_obj, *out_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOssiiOOOOOOOOO", &values_obj, &thresholds_obj, &features_obj, &children_left_obj, &children_right_obj, &children_missing_obj, &cat_values_obj, &cat_start_obj, &cat_size_obj, &tree_offsets_obj, &cat_offsets_obj, &reference_data_obj, &explain_point_obj, &decision_type_cptr, &index_cptr, &max_order, &verbose, &weight_table_obj, &subset_keys_obj, &subset_starts_obj, &counts_obj, &slot_keys_obj, &slot_rows_obj, &block_starts_obj, &block_shifts_obj, &out_obj))
        return NULL;
    if (max_order < 1)
    {
        PyErr_SetString(PyExc_ValueError, "max_order must be >= 1");
        return NULL;
    }
    IndexType index_type;
    if (!parse_index_type(std::string(index_cptr), index_type))
    {
        PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + std::string(index_cptr)).c_str());
        return NULL;
    }
    std::vector<PyArrayObject *> arrays_for_decref;
    auto cleanup_arrays = [&arrays_for_decref]()
    {
        for (PyArrayObject *arr : arrays_for_decref)
            Py_XDECREF(arr);
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
    EnsembleBlock block;
    if (!block.parse(values_obj, thresholds_obj, features_obj, children_left_obj, children_right_obj, children_missing_obj, cat_values_obj, cat_start_obj, cat_size_obj, tree_offsets_obj, cat_offsets_obj, decision_type_cptr, n_features))
    {
        cleanup_arrays();
        return NULL;
    }
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
    return compute_interactions_structural(
        block.trees, reference_data, explain_data, n_ref, n_features, index_type, max_order,
        custom_table, custom_N, custom_K, subset_keys_obj, subset_starts_obj, counts_obj,
        slot_keys_obj, slot_rows_obj, block_starts_obj, block_shifts_obj, out_obj, cleanup_arrays);
}

// === predict_ensemble_sum ===
// C port of shapiq.tree.base.predict_ensemble: route every row of X through
// every tree (same goes_left semantics as the interaction kernels — NaN routing,
// categorical splits, decision type) and return the per-row sum of leaf values.
// Used for the interventional baseline_value, which was a Python while loop
// over every (row, tree) pair.
static PyObject *predict_ensemble_sum(PyObject *self, PyObject *args)
{
    PyObject *values_obj, *thresholds_obj, *features_obj, *children_left_obj, *children_right_obj,
        *children_missing_obj, *cat_values_obj, *cat_start_obj, *cat_size_obj, *tree_offsets_obj,
        *cat_offsets_obj;
    PyObject *x_data_obj;
    const char *decision_type_cptr;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOs", &values_obj, &thresholds_obj, &features_obj, &children_left_obj, &children_right_obj, &children_missing_obj, &cat_values_obj, &cat_start_obj, &cat_size_obj, &tree_offsets_obj, &cat_offsets_obj, &x_data_obj, &decision_type_cptr))
        return NULL;
    std::vector<PyArrayObject *> arrays_for_decref;
    auto cleanup_arrays = [&arrays_for_decref]()
    {
        for (PyArrayObject *arr : arrays_for_decref)
            Py_XDECREF(arr);
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
    const npy_intp n_rows = PyArray_DIM(x_data_array, 0);
    const int n_features = (int)PyArray_DIM(x_data_array, 1);
    const double *x_data = (const double *)PyArray_DATA(x_data_array);
    EnsembleBlock block;
    if (!block.parse(values_obj, thresholds_obj, features_obj, children_left_obj, children_right_obj, children_missing_obj, cat_values_obj, cat_start_obj, cat_size_obj, tree_offsets_obj, cat_offsets_obj, decision_type_cptr, n_features))
    {
        cleanup_arrays();
        return NULL;
    }
    std::vector<Tree> &trees = block.trees;

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

// === preprocess_subset_tables ===
// Structural DFS over every root-to-leaf path of every tree, collecting each feature subset
// (orders 2..max_order) that co-occurs on some path: the only interactions an explanation can
// touch, since a leaf's E and R are always drawn from its path. A subset is emitted at the
// first occurrence of its deepest member (a feature repeated on a path adds nothing new), and
// both children are always walked -- which child an explain/reference pair takes is unknown
// here. Rows are keyed as base-n_features integers while collecting (sorted features as
// digits); the key must fit in an int64, else None is returned and the caller falls back.
// Returns (keys, starts, counts) in the layout of shapiq.tree.subset_index.subset_table_keys.
namespace
{
    // Open-addressing set of int64 keys (multiply-shift hash, linear probing, <= 1/2 full).
    struct FlatKeySet
    {
        std::vector<int64_t> slots;
        uint64_t mask = 0;
        int shift = 63;
        size_t size = 0;

        FlatKeySet() { reset(1024); }

        void reset(size_t capacity)
        {
            int bits = 1;
            while ((size_t(1) << bits) < capacity)
                ++bits;
            slots.assign(size_t(1) << bits, -1);
            mask = (uint64_t(1) << bits) - 1;
            shift = 64 - bits;
            size = 0;
        }

        void insert(int64_t key)
        {
            uint64_t slot = (static_cast<uint64_t>(key) * algorithms::kIndexHashMultiplier) >> shift;
            while (true)
            {
                const int64_t stored = slots[slot];
                if (stored == key)
                    return;
                if (stored < 0)
                {
                    if (2 * (size + 1) > slots.size())
                    {
                        grow();
                        insert(key);
                        return;
                    }
                    slots[slot] = key;
                    ++size;
                    return;
                }
                slot = (slot + 1) & mask;
            }
        }

        void grow()
        {
            std::vector<int64_t> old = std::move(slots);
            reset(old.size() * 2);
            for (int64_t key : old)
                if (key >= 0)
                    insert(key);
        }

        std::vector<int64_t> sorted_keys() const
        {
            std::vector<int64_t> keys;
            keys.reserve(size);
            for (int64_t key : slots)
                if (key >= 0)
                    keys.push_back(key);
            std::sort(keys.begin(), keys.end());
            return keys;
        }
    };

    struct SubsetCollector
    {
        int n_features;
        int max_order;
        std::vector<int> path;                        // sorted distinct features on the path
        std::vector<FlatKeySet> keys;                 // per order: distinct keys of emitted subsets
        std::vector<int> chosen;

        void emit(int new_feature)
        {
            // every (order-1)-subset of the current path, merged with new_feature
            for (int order = 2; order <= max_order; ++order)
            {
                if (static_cast<int>(path.size()) < order - 1)
                    break;
                choose(order - 1, 0, 0, new_feature, keys[order]);
            }
        }

        void choose(int remaining, int start, int level, int new_feature, FlatKeySet &out)
        {
            if (remaining == 0)
            {
                int64_t key = 0;
                int taken = 0;
                bool placed = false;
                for (int i = 0; i < level + 1; ++i)
                {
                    int feature;
                    if (!placed && (taken == level || chosen[taken] > new_feature))
                    {
                        feature = new_feature;
                        placed = true;
                    }
                    else
                        feature = chosen[taken++];
                    key = key * n_features + feature;
                }
                out.insert(key);
                return;
            }
            const int n = static_cast<int>(path.size());
            for (int i = start; i <= n - remaining; ++i)
            {
                chosen[level] = path[i];
                choose(remaining - 1, i + 1, level + 1, new_feature, out);
            }
        }

        void walk(const int64_t *features, const int64_t *cl, const int64_t *cr, int64_t node)
        {
            if (cl[node] == cr[node])  // leaf (both -1), as the explanation kernels test it
                return;
            const int feature = static_cast<int>(features[node]);
            auto at = std::lower_bound(path.begin(), path.end(), feature);
            const bool first = (at == path.end() || *at != feature);
            if (first)
            {
                emit(feature);
                path.insert(at, feature);
            }
            walk(features, cl, cr, cl[node]);
            walk(features, cl, cr, cr[node]);
            if (first)
                path.erase(std::lower_bound(path.begin(), path.end(), feature));
        }
    };
}

static PyObject *preprocess_subset_tables(PyObject *self, PyObject *args)
{
    PyObject *features_obj, *children_left_obj, *children_right_obj, *tree_offsets_obj;
    int n_features;
    int max_order;
    if (!PyArg_ParseTuple(args, "OOOOii", &features_obj, &children_left_obj, &children_right_obj,
                          &tree_offsets_obj, &n_features, &max_order))
        return NULL;
    if (n_features < 1 || max_order < 1)
    {
        PyErr_SetString(PyExc_ValueError, "n_features and max_order must be >= 1");
        return NULL;
    }
    // the base-n_features key of a max_order subset must fit in an int64
    {
        int64_t cap = 1;
        for (int i = 0; i < max_order; ++i)
        {
            if (cap > INT64_MAX / n_features)
                Py_RETURN_NONE;
            cap *= n_features;
        }
    }
    PyArrayObject *feat_arr = (PyArrayObject *)PyArray_FROM_OTF(features_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cl_arr = (PyArrayObject *)PyArray_FROM_OTF(children_left_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cr_arr = (PyArrayObject *)PyArray_FROM_OTF(children_right_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *toff_arr = (PyArrayObject *)PyArray_FROM_OTF(tree_offsets_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    auto release = [&]()
    {
        Py_XDECREF(feat_arr);
        Py_XDECREF(cl_arr);
        Py_XDECREF(cr_arr);
        Py_XDECREF(toff_arr);
    };
    if (!feat_arr || !cl_arr || !cr_arr || !toff_arr || PyArray_NDIM(feat_arr) != 1 || PyArray_NDIM(cl_arr) != 1 ||
        PyArray_NDIM(cr_arr) != 1 || PyArray_NDIM(toff_arr) != 1)
    {
        release();
        PyErr_SetString(PyExc_TypeError, "features, children_left, children_right and tree_offsets must be 1-D int64 numpy arrays");
        return NULL;
    }
    const npy_intp n_nodes = PyArray_DIM(feat_arr, 0);
    const int64_t n_trees = (int64_t)PyArray_DIM(toff_arr, 0) - 1;
    const int64_t *features = (const int64_t *)PyArray_DATA(feat_arr);
    const int64_t *cl = (const int64_t *)PyArray_DATA(cl_arr);
    const int64_t *cr = (const int64_t *)PyArray_DATA(cr_arr);
    const int64_t *toff = (const int64_t *)PyArray_DATA(toff_arr);
    bool valid = PyArray_DIM(cl_arr, 0) == n_nodes && PyArray_DIM(cr_arr, 0) == n_nodes && n_trees >= 1 &&
                 toff[0] == 0 && toff[n_trees] == n_nodes;
    for (int64_t t = 0; valid && t < n_trees; ++t)
    {
        const int64_t off = toff[t], n_t = toff[t + 1] - off;
        if (n_t < 1)
            valid = false;
        for (int64_t i = off; valid && i < off + n_t; ++i)
        {
            if (cl[i] == cr[i])
                continue;
            if (features[i] < 0 || features[i] >= n_features || cl[i] < 0 || cl[i] >= n_t || cr[i] < 0 || cr[i] >= n_t)
                valid = false;
        }
    }
    if (!valid)
    {
        release();
        PyErr_SetString(PyExc_ValueError, "tree arrays are inconsistent (offsets, feature or child index out of range)");
        return NULL;
    }

    SubsetCollector collector;
    collector.n_features = n_features;
    collector.max_order = max_order;
    collector.keys.resize(static_cast<size_t>(max_order) + 1);
    collector.chosen.resize(static_cast<size_t>(max_order));
    for (int64_t t = 0; t < n_trees; ++t)
        collector.walk(features + toff[t], cl + toff[t], cr + toff[t], 0);
    release();

    // per order: the distinct keys in lexicographic (= numeric) order are the table rows;
    // the hash index maps each key to its row (layout of shapiq.tree.subset_index)
    npy_intp n_orders = max_order + 1;
    std::vector<std::vector<int64_t>> sorted(static_cast<size_t>(n_orders));
    int64_t total = 0, n_slots = 0;
    std::vector<int> bits(static_cast<size_t>(n_orders), 1);
    for (int order = 2; order <= max_order; ++order)
    {
        sorted[order] = collector.keys[order].sorted_keys();
        const int64_t count = static_cast<int64_t>(sorted[order].size());
        if (count > INT32_MAX)
        {
            PyErr_SetString(PyExc_ValueError, "subset tables beyond 2^31 rows per order are not supported.");
            return NULL;
        }
        total += count * order;
        while ((int64_t(1) << bits[order]) < 2 * count)  // block at most half full
            ++bits[order];
        n_slots += int64_t(1) << bits[order];
    }
    PyObject *starts = PyArray_ZEROS(1, &n_orders, NPY_INT64, 0);
    PyObject *counts = PyArray_ZEROS(1, &n_orders, NPY_INT64, 0);
    PyObject *block_starts = PyArray_ZEROS(1, &n_orders, NPY_INT64, 0);
    PyObject *block_shifts = PyArray_ZEROS(1, &n_orders, NPY_INT64, 0);
    npy_intp n_total = total, n_slots_np = n_slots;
    PyObject *keys = PyArray_SimpleNew(1, &n_total, NPY_INT32);
    PyObject *slot_keys = PyArray_SimpleNew(1, &n_slots_np, NPY_INT64);
    PyObject *slot_rows = PyArray_SimpleNew(1, &n_slots_np, NPY_INT32);
    if (!starts || !counts || !block_starts || !block_shifts || !keys || !slot_keys || !slot_rows)
    {
        Py_XDECREF(starts);
        Py_XDECREF(counts);
        Py_XDECREF(block_starts);
        Py_XDECREF(block_shifts);
        Py_XDECREF(keys);
        Py_XDECREF(slot_keys);
        Py_XDECREF(slot_rows);
        return NULL;
    }
    int64_t *starts_p = (int64_t *)PyArray_DATA((PyArrayObject *)starts);
    int64_t *counts_p = (int64_t *)PyArray_DATA((PyArrayObject *)counts);
    int64_t *bstarts_p = (int64_t *)PyArray_DATA((PyArrayObject *)block_starts);
    int64_t *bshifts_p = (int64_t *)PyArray_DATA((PyArrayObject *)block_shifts);
    int32_t *out = (int32_t *)PyArray_DATA((PyArrayObject *)keys);
    int64_t *skeys = (int64_t *)PyArray_DATA((PyArrayObject *)slot_keys);
    int32_t *srows = (int32_t *)PyArray_DATA((PyArrayObject *)slot_rows);
    std::fill(skeys, skeys + n_slots, int64_t(-1));
    std::fill(srows, srows + n_slots, int32_t(-1));
    std::fill(bshifts_p, bshifts_p + n_orders, int64_t(63));
    int64_t position = 0, slot_position = 0;
    for (int order = 2; order <= max_order; ++order)
    {
        const auto &rows = sorted[order];
        starts_p[order] = position;
        counts_p[order] = static_cast<int64_t>(rows.size());
        bstarts_p[order] = slot_position;
        bshifts_p[order] = 64 - bits[order];
        const uint64_t mask = (uint64_t(1) << bits[order]) - 1;
        int64_t *block_keys = skeys + slot_position;
        int32_t *block_rows = srows + slot_position;
        for (size_t r = 0; r < rows.size(); ++r)
        {
            int64_t key = rows[r];
            uint64_t slot = (static_cast<uint64_t>(key) * algorithms::kIndexHashMultiplier) >> (64 - bits[order]);
            while (block_keys[slot] >= 0)
                slot = (slot + 1) & mask;
            block_keys[slot] = key;
            block_rows[slot] = static_cast<int32_t>(r);
            for (int d = order - 1; d >= 0; --d)
            {
                out[d] = static_cast<int32_t>(key % n_features);
                key /= n_features;
            }
            out += order;
        }
        position += static_cast<int64_t>(rows.size()) * order;
        slot_position += int64_t(1) << bits[order];
    }
    return Py_BuildValue("((NNN)(NNNN))", keys, starts, counts, slot_keys, slot_rows, block_starts, block_shifts);
}
