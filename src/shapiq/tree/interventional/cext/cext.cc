// Python bindings of the interventional tree kernels.
//
// Two fills produce leaf events (cohort_fill.hpp over a background dataset,
// single_reference_fill.hpp over one reference row), one leaf update adds them to the output
// array (leaf_update.hpp), whose layout comes from ../../cext/subset_tables.hpp.
//
//   preprocess_subset_tables(features, children_left, children_right, tree_offsets,
//                            n_features, max_order[, complete_above])
//       -> ((keys, counts), (slot_keys, slot_rows) or None)
//   fill_cohort(<ensemble block: 11 arrays>, reference_data (n_ref x n_features), x (n_features),
//               decision_type, index, max_order, custom_weight_table or None, keys, counts,
//               slot_keys, slot_rows (both None without an index: the tables are searched), out)
//   fill_single_reference(<ensemble block>, reference_row (n_features), x, ... as above ...)
//   layout_to_dict(out, keys, counts, n_features, min_order, feature_ids, skip_zeros)
//   predict_ensemble_sum(<ensemble block>, X (n_rows x n_features), decision_type) -> (n_rows)
//
// Only cext.cc is listed in setup.py; an edit of an included file needs `rm -rf build`.
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <memory>
#include <cstdint>
#include <string>
#include <vector>

#include <omp.h>

#include "utils.cpp"
#include "weights.cpp"
#include "leaf_update.hpp"
#include "single_reference_fill.hpp"
#include "cohort_fill.hpp"

static PyObject *fill_cohort(PyObject *self, PyObject *args);
static PyObject *fill_single_reference(PyObject *self, PyObject *args);
static PyObject *predict_ensemble_sum(PyObject *self, PyObject *args);
static PyObject *preprocess_subset_tables(PyObject *self, PyObject *args);
static PyObject *layout_to_dict(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"fill_cohort", fill_cohort, METH_VARARGS,
     "Cohort fill: one DFS per tree over all reference rows, adding every order into the float64 output array `out`."},
    {"fill_single_reference", fill_single_reference, METH_VARARGS,
     "Single-reference fill: the interventional recursion for one reference row, same output array as fill_cohort."},
    {"predict_ensemble_sum", predict_ensemble_sum, METH_VARARGS,
     "Route every row of X through every tree and return the per-row sum of leaf values."},
    {"preprocess_subset_tables", preprocess_subset_tables, METH_VARARGS,
     "Collect per order the interactions whose feature ids all occur on one root-to-leaf path, with the hash index from interaction to position: ((keys, counts), (slot_keys, slot_rows) or None when a key would overflow 127 bits)."},
    {"layout_to_dict", layout_to_dict, METH_VARARGS,
     "Read a kernel's output array back into {feature tuple: value}."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cext",
    "Interventional Shapley-interaction kernels for tree ensembles.",
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

static bool parse_index_type(const std::string &index, IndexType &index_type)
{
    if (index == "SII" || index == "SV")
        index_type = IndexType::SII;
    else if (index == "BII" || index == "BV")
        index_type = IndexType::BII;
    else if (index == "CHII" || index == "CV")
        index_type = IndexType::CHII;
    else if (index == "FBII")
        index_type = IndexType::FBII;
    else if (index == "FSII")
        index_type = IndexType::FSII;
    else if (index == "STII")
        index_type = IndexType::STII;
    else if (index == "Moebius")
        index_type = IndexType::MOEBIUS;
    else if (index == "Fourier")
        index_type = IndexType::FOURIER;
    else if (index == "CUSTOM")
        index_type = IndexType::CUSTOM;
    else
        return false;
    return true;
}

// === the ensemble block ===
// All trees' node arrays back to back with tree-relative child ids (built by
// InterventionalTreeSHAPIQ._ensemble_block): tree t owns nodes [tree_offsets[t],
// tree_offsets[t + 1]) and categorical values [cat_offsets[t], cat_offsets[t + 1]).
struct EnsembleBlock
{
    std::vector<PyArrayObject *> owned;
    std::vector<Tree> trees;
    int64_t n_nodes = 0;

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

    // Parses and validates the 11 arrays; n_features < 0 skips the split-feature range check.
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
        const int64_t *features = (const int64_t *)PyArray_DATA(feats);
        const int64_t *children_left = (const int64_t *)PyArray_DATA(cl);
        const int64_t *children_right = (const int64_t *)PyArray_DATA(cr);
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

// Longest root-to-leaf path in internal nodes (a feature may repeat on a path).
static int longest_path(const Tree &tree)
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

// === shared argument handling of the two fills ===
// Thread-private output arrays are used while they total at most this many bytes; above it
// the threads add atomically into the shared output array.
static const size_t kMaxThreadBlockBytes = size_t(32) << 20;

struct FillCall
{
    EnsembleBlock block;
    std::vector<PyArrayObject *> arrays;
    const double *reference = nullptr; // n_ref x n_features
    const double *x = nullptr;         // the explained point, n_features
    int n_ref = 0;
    int n_features = 0;
    int max_order = 0;
    IndexType index_type = IndexType::SII;
    const double *custom_table = nullptr;
    const int64_t *counts = nullptr;
    subset_tables::Geometry geometry;
    interventional::Layout layout;
    std::vector<int64_t> rank3_base;
    double *out = nullptr;
    PyArrayObject *out_arr = nullptr;
    int max_depth = 0; // longest root-to-leaf path in internal nodes: sizes the path scratch
    int max_width = 0; // bound on |E| + |R| (distinct features on a path): sizes the weight table

    ~FillCall()
    {
        if (out_arr)
            PyArray_ResolveWritebackIfCopy(out_arr);
        for (PyArrayObject *arr : arrays)
            Py_XDECREF(arr);
    }

    PyArrayObject *take(PyObject *obj, int np_type, int flags = NPY_ARRAY_IN_ARRAY)
    {
        PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(obj, np_type, flags);
        if (arr)
            arrays.push_back(arr);
        return arr;
    }

    // Parses the positional arguments listed in the file header; `reference_ndim` is 2 for the
    // cohort fill and 1 for the single-reference fill. Sets a Python error on failure. An index
    // that is given is trusted; without one the leaf update searches the tables.
    bool parse(PyObject *args, int reference_ndim)
    {
        PyObject *ens[11];
        PyObject *reference_obj, *x_obj, *weight_table_obj, *keys_obj, *counts_obj, *slot_keys_obj, *slot_rows_obj, *out_obj;
        const char *decision_type_cptr, *index_cptr;
        if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOssiOOOOOO", &ens[0], &ens[1], &ens[2], &ens[3], &ens[4], &ens[5], &ens[6],
                              &ens[7], &ens[8], &ens[9], &ens[10], &reference_obj, &x_obj, &decision_type_cptr, &index_cptr,
                              &max_order, &weight_table_obj, &keys_obj, &counts_obj, &slot_keys_obj, &slot_rows_obj, &out_obj))
            return false;
        if (max_order < 1)
        {
            PyErr_SetString(PyExc_ValueError, "max_order must be >= 1");
            return false;
        }
        if (!parse_index_type(std::string(index_cptr), index_type))
        {
            PyErr_SetString(PyExc_ValueError, ("Unsupported index type: " + std::string(index_cptr)).c_str());
            return false;
        }
        PyArrayObject *reference_arr = take(reference_obj, NPY_FLOAT64);
        PyArrayObject *x_arr = take(x_obj, NPY_FLOAT64);
        if (!reference_arr || !x_arr || PyArray_NDIM(reference_arr) != reference_ndim || PyArray_NDIM(x_arr) != 1)
        {
            PyErr_SetString(PyExc_TypeError, reference_ndim == 2
                                                 ? "reference_data must be a 2-D float64 numpy array and x a 1-D float64 numpy array"
                                                 : "reference_row and x must be 1-D float64 numpy arrays");
            return false;
        }
        n_ref = reference_ndim == 2 ? (int)PyArray_DIM(reference_arr, 0) : 1;
        n_features = (int)PyArray_DIM(reference_arr, reference_ndim - 1);
        if ((int)PyArray_DIM(x_arr, 0) != n_features || n_ref < 1)
        {
            PyErr_SetString(PyExc_ValueError, "x must have one entry per feature of the reference data");
            return false;
        }
        reference = (const double *)PyArray_DATA(reference_arr);
        x = (const double *)PyArray_DATA(x_arr);
        if (!block.parse(ens[0], ens[1], ens[2], ens[3], ens[4], ens[5], ens[6], ens[7], ens[8], ens[9], ens[10],
                         decision_type_cptr, n_features))
            return false;
        if (weight_table_obj != Py_None)
        {
            PyArrayObject *weight_arr = take(weight_table_obj, NPY_FLOAT64);
            if (!weight_arr)
            {
                PyErr_SetString(PyExc_TypeError, "weight_table must be a float64 numpy array");
                return false;
            }
            custom_table = (const double *)PyArray_DATA(weight_arr);
        }
        if (index_type == IndexType::CUSTOM && !custom_table)
        {
            PyErr_SetString(PyExc_ValueError, "index CUSTOM needs a weight table");
            return false;
        }
        PyArrayObject *keys_arr = take(keys_obj, NPY_INT32);
        PyArrayObject *counts_arr = take(counts_obj, NPY_INT64);
        const bool has_index = slot_keys_obj != Py_None;
        PyArrayObject *slot_keys_arr = has_index ? take(slot_keys_obj, NPY_INT64) : NULL;
        PyArrayObject *slot_rows_arr = has_index ? take(slot_rows_obj, NPY_INT32) : NULL;
        out_arr = take(out_obj, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY2);
        if (!keys_arr || !counts_arr || !out_arr || (has_index && (!slot_keys_arr || !slot_rows_arr)))
        {
            PyErr_SetString(PyExc_TypeError, "layout arrays must be numpy arrays (int32 keys, int64 counts, float64 out; int64 slot keys and int32 slot rows, or both None)");
            return false;
        }
        const char *error = NULL;
        if (PyArray_NDIM(keys_arr) != 1 || PyArray_NDIM(counts_arr) != 1 || PyArray_NDIM(out_arr) != 1 ||
            (has_index && (PyArray_NDIM(slot_keys_arr) != 1 || PyArray_NDIM(slot_rows_arr) != 1)))
            error = "layout arrays must be 1-dimensional.";
        else if (PyArray_DIM(counts_arr, 0) != max_order + 1 ||
                 (has_index && 2 * PyArray_DIM(slot_rows_arr, 0) != PyArray_DIM(slot_keys_arr, 0)))
            error = "layout arrays have inconsistent lengths (slot_keys holds two words per slot).";
        if (!error)
        {
            counts = (const int64_t *)PyArray_DATA(counts_arr);
            for (int order = 2; !error && order <= max_order; ++order)
                if (counts[order] < 0)
                    error = "subset counts must be non-negative.";
        }
        if (!error)
        {
            geometry = subset_tables::derive_geometry(counts, max_order, n_features, 1);
            error = subset_tables::check_layout(geometry, (int64_t)PyArray_DIM(keys_arr, 0),
                                                has_index ? (int64_t)PyArray_DIM(slot_rows_arr, 0) : -1,
                                                (int64_t)PyArray_DIM(out_arr, 0));
        }
        if (error)
        {
            PyErr_SetString(PyExc_ValueError, error);
            return false;
        }
        layout.n_features = n_features;
        layout.max_order = max_order;
        layout.counts = counts;
        layout.order_offsets = geometry.output_offsets.data();
        layout.keys = (const int32_t *)PyArray_DATA(keys_arr);
        layout.table_starts = geometry.table_starts.data();
        layout.has_index = has_index;
        if (has_index)
        {
            layout.index.keys = (const int64_t *)PyArray_DATA(slot_keys_arr);
            layout.index.rows = (const int32_t *)PyArray_DATA(slot_rows_arr);
            layout.index.starts = geometry.block_starts.data();
            layout.index.shifts = geometry.block_shifts.data();
        }
        // a complete order-2/3 table is addressed by lexicographic rank, not through the index
        const int64_t nf = n_features;
        layout.dense2 = max_order >= 2 && subset_tables::table_is_complete(counts[2], 2, nf);
        layout.dense3 = max_order >= 3 && subset_tables::table_is_complete(counts[3], 3, nf);
        if (layout.dense3)
        {
            rank3_base = interventional::make_rank3_base(nf);
            layout.rank3_base = rank3_base.data();
        }
        out = (double *)PyArray_DATA(out_arr);
        for (const Tree &tree : block.trees)
            max_depth = std::max(max_depth, longest_path(tree));
        max_width = std::min(max_depth, n_features); // a feature may repeat on a path
        return true;
    }
};

// Runs `per_tree(tree, update, tid)` over all trees in parallel. One thread writes straight
// into `out`; several threads write thread-private arrays that are summed into `out` afterwards
// (serially while small, in parallel once large), or add atomically into `out` when the private
// arrays would exceed kMaxThreadBlockBytes. Returns 1 when an interaction was missing from its
// table, 2 when no leaf update matched the layout (an internal error), else 0.
template <typename PerTree>
static int run_fill(FillCall &call, PerTree &&per_tree)
{
    const int stride = call.max_width + 1;
    const std::vector<double> weights = inter_weights::build_weight_table(
        call.index_type, call.n_features, call.max_order, call.max_width, call.custom_table,
        (int64_t)call.n_features + 1, (int64_t)call.max_order + 1);
    const int64_t n_rows = call.geometry.n_outputs;
    const int n_threads = omp_get_max_threads();
    const bool direct = n_threads == 1;
    const bool atomic = !direct && static_cast<size_t>(n_threads) * static_cast<size_t>(n_rows) * sizeof(double) > kMaxThreadBlockBytes;
    const bool parallel_merge = !direct && !atomic && static_cast<size_t>(n_threads) * static_cast<size_t>(n_rows) > (size_t(1) << 16);
    std::vector<double> block((direct || atomic) ? 0 : static_cast<size_t>(n_threads) * static_cast<size_t>(n_rows), 0.0);
    std::vector<int> thread_status(static_cast<size_t>(n_threads), 0);
    const Py_ssize_t num_trees = (Py_ssize_t)call.block.trees.size();
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double *target = (direct || atomic) ? call.out : block.data() + static_cast<size_t>(tid) * n_rows;
        const int status = interventional::dispatch_leaf_update(
            call.layout, weights.data(), stride, target, atomic, [&](auto &update)
            {
                if (parallel_merge)
                {
#pragma omp for schedule(dynamic, 1)
                    for (Py_ssize_t t = 0; t < num_trees; t++)
                        per_tree(call.block.trees[t], update, tid);
                }
                else
                {
#pragma omp for schedule(dynamic, 1) nowait
                    for (Py_ssize_t t = 0; t < num_trees; t++)
                        per_tree(call.block.trees[t], update, tid);
                }
            });
        thread_status[static_cast<size_t>(tid)] = status;
        if (parallel_merge)
        {
            // every thread's private array is complete here (the loop above ends with a barrier)
#pragma omp for schedule(static)
            for (int64_t k = 0; k < n_rows; k++)
            {
                double acc = 0.0;
                for (int th = 0; th < n_threads; th++)
                    acc += block[static_cast<size_t>(th) * n_rows + k];
                call.out[k] += acc;
            }
        }
        else if (!direct && !atomic)
        {
#pragma omp critical
            {
                for (int64_t k = 0; k < n_rows; k++)
                    call.out[k] += target[k];
            }
        }
    }
    int status = 0;
    for (int flag : thread_status)
        status = std::max(status, flag);
    return status;
}

static PyObject *finish_fill(int status)
{
    if (status == 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "an interaction reached by the walk is missing from the interaction tables (preprocess_subset_tables and the fill disagree).");
        return NULL;
    }
    if (status != 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "no leaf update matches this layout (internal error in dispatch_leaf_update).");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *fill_cohort(PyObject *self, PyObject *args)
{
    (void)self;
    FillCall call;
    if (!call.parse(args, 2))
        return NULL;
    int status = 0;
    Py_BEGIN_ALLOW_THREADS
    const double scale = 1.0 / static_cast<double>(call.n_ref);
    // one scratch per thread, allocated on first use and freed with the call
    std::vector<std::unique_ptr<interventional::CohortScratch>> scratch(static_cast<size_t>(omp_get_max_threads()));
    status = run_fill(call, [&](Tree &tree, auto &update, int tid)
                      {
        std::unique_ptr<interventional::CohortScratch> &s = scratch[static_cast<size_t>(tid)];
        if (!s)
            s.reset(new interventional::CohortScratch(call.max_depth, call.n_features, call.n_ref));
        interventional::cohort_fill(tree, call.reference, call.x, call.n_ref, call.n_features, scale, *s, update); });
    Py_END_ALLOW_THREADS
    return finish_fill(status);
}

static PyObject *fill_single_reference(PyObject *self, PyObject *args)
{
    (void)self;
    FillCall call;
    if (!call.parse(args, 1))
        return NULL;
    int status = 0;
    Py_BEGIN_ALLOW_THREADS
    std::vector<std::unique_ptr<interventional::PathScratch>> paths(static_cast<size_t>(omp_get_max_threads()));
    std::vector<std::vector<interventional::SingleFrame>> stacks(static_cast<size_t>(omp_get_max_threads()));
    status = run_fill(call, [&](Tree &tree, auto &update, int tid)
                      {
        std::unique_ptr<interventional::PathScratch> &p = paths[static_cast<size_t>(tid)];
        if (!p)
            p.reset(new interventional::PathScratch(call.max_depth));
        interventional::single_reference_fill(tree, call.x, call.reference, 1.0, *p, stacks[static_cast<size_t>(tid)], update); });
    Py_END_ALLOW_THREADS
    return finish_fill(status);
}

// === predict_ensemble_sum ===
// Routes every row of X through every tree with the kernels' goes_left (NaN routing,
// categorical splits, decision type) and returns the per-row sum of leaf values.
static PyObject *predict_ensemble_sum(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *ens[11];
    PyObject *x_data_obj;
    const char *decision_type_cptr;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOs", &ens[0], &ens[1], &ens[2], &ens[3], &ens[4], &ens[5], &ens[6], &ens[7],
                          &ens[8], &ens[9], &ens[10], &x_data_obj, &decision_type_cptr))
        return NULL;
    PyArrayObject *x_data_array = (PyArrayObject *)PyArray_FROM_OTF(x_data_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!x_data_array || PyArray_NDIM(x_data_array) != 2)
    {
        Py_XDECREF(x_data_array);
        PyErr_SetString(PyExc_TypeError, "X must be a 2-D float64 numpy array");
        return NULL;
    }
    const npy_intp n_rows = PyArray_DIM(x_data_array, 0);
    const int n_features = (int)PyArray_DIM(x_data_array, 1);
    const double *x_data = (const double *)PyArray_DATA(x_data_array);
    EnsembleBlock block;
    if (!block.parse(ens[0], ens[1], ens[2], ens[3], ens[4], ens[5], ens[6], ens[7], ens[8], ens[9], ens[10],
                     decision_type_cptr, n_features))
    {
        Py_DECREF(x_data_array);
        return NULL;
    }
    std::vector<Tree> &trees = block.trees;
    npy_intp out_dim = n_rows;
    PyObject *out_array = PyArray_SimpleNew(1, &out_dim, NPY_FLOAT64);
    if (!out_array)
    {
        Py_DECREF(x_data_array);
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
                node = tree.goes_left(row[feature], node) ? tree.children_left[node] : tree.children_right[node];
            }
            total += tree.leaf_predictions[node];
        }
        out[i] = total;
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(x_data_array);
    return out_array;
}

// preprocess_subset_tables and layout_to_dict are defined in ../../cext/subset_tables.hpp.
