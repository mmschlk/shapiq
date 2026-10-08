// Co-occurring feature subset tables and their hash index, shared by the tree kernels.
//
// A kernel that computes interactions only over the feature subsets that can be non-zero --
// those whose features co-occur on some root-to-leaf path -- keeps, per order >= 2, a sorted
// table of those subsets and needs to map an enumerated subset to its row. This header holds
// everything the interventional and the quadrature extension need for that:
//   * the integer key of a sorted subset (its features as digits in base n_features) and the
//     multiply-shift hash of that key,
//   * Geometry: every offset of the layout, derived from the per-order row counts alone,
//   * SubsetIndex, the flat open-addressing index the kernels probe,
//   * SubsetCollector, one structural DFS per tree collecting the tables and the index, and
//   * the Python bindings `preprocess_subset_tables` and `layout_to_dict`, registered by both
//     extension modules: the former builds the layout, the latter reads a kernel's output
//     array back into {feature tuple: value}.
//
// The layout. The output array is the order-1 block [0, n_features) (when min_order == 1)
// followed, per order s >= 2, by one entry per row of order s's table, rows in lexicographic
// order. The index holds, per order, a block of 2^block_bits(count) slots of (key, row), -1 an
// empty slot; a probe starts at the top bits of key * kIndexHashMultiplier and steps linearly.
// Table starts, block starts and sizes and output offsets all follow from `counts`, so the
// kernels receive only (keys, counts, slot_keys, slot_rows) and derive the rest here.
//
// Included from compiled extension sources only. setuptools does not track included files:
// after editing, rebuild with `rm -rf build && uv run python setup.py build_ext --inplace`.
// The binding at the end needs <Python.h> and <numpy/arrayobject.h> included before this file.
// The numpy oracle in tests/.../subset_index_reference.py mirrors the key, hash and
// block_bits rule; a change here must be mirrored there.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

namespace subset_tables
{
    // 2^64 / golden ratio; the reference builder in the tests uses the same constant.
    constexpr uint64_t kIndexHashMultiplier = 0x9E3779B97F4A7C15ULL;

    // Whether n_features^max_order fits in an int64, i.e. whether the key encoding exists.
    inline bool keys_fit(int64_t n_features, int max_order)
    {
        int64_t cap = 1;
        for (int i = 0; i < max_order; ++i)
        {
            if (cap > INT64_MAX / n_features)
                return false;
            cap *= n_features;
        }
        return true;
    }

    // log2 of the index block size for a table of `count` rows: the smallest power of two
    // holding at least 2 * count slots (the block stays at most half full), at least 2 slots.
    inline int block_bits(int64_t count)
    {
        int bits = 1;
        while ((int64_t(1) << bits) < 2 * count)
            ++bits;
        return bits;
    }

    // Every offset of the layout, derived from the per-order counts (one entry per order,
    // index 0 and 1 unused for tables).
    struct Geometry
    {
        std::vector<int64_t> table_starts;    // per order: position of its table in `keys`
        std::vector<int64_t> block_starts;    // per order: first slot of its index block
        std::vector<int64_t> block_shifts;    // per order: 64 - block_bits(count)
        std::vector<int64_t> output_offsets;  // per order: start of its block in the output
        int64_t n_keys = 0;                   // length of `keys`
        int64_t n_slots = 0;                  // length of the index arrays
        int64_t n_outputs = 0;                // length of the output array
    };

    // `counts` has max_order + 1 entries; orders below max(min_order, 2) own no output block
    // (their tables and index blocks still exist, so the probe geometry is order-independent).
    inline Geometry derive_geometry(const int64_t *counts, int max_order, int64_t n_features, int min_order)
    {
        Geometry g;
        const size_t n_orders = static_cast<size_t>(max_order) + 1;
        g.table_starts.assign(n_orders, 0);
        g.block_starts.assign(n_orders, 0);
        g.block_shifts.assign(n_orders, 63);
        g.output_offsets.assign(n_orders, 0);
        g.n_outputs = (min_order <= 1) ? n_features : 0;
        for (int order = 2; order <= max_order; ++order)
        {
            const int64_t count = counts[order];
            g.table_starts[order] = g.n_keys;
            g.n_keys += count * order;
            const int bits = block_bits(count);
            g.block_starts[order] = g.n_slots;
            g.block_shifts[order] = 64 - bits;
            g.n_slots += int64_t(1) << bits;
            if (order >= min_order)
            {
                g.output_offsets[order] = g.n_outputs;
                g.n_outputs += count;
            }
        }
        return g;
    }

    // Key of the sorted tuple obtained by splicing `feature` into the sorted `chosen[0, size)`
    // (which does not contain it); no tuple is materialized. A fixed size + 1 iterations rather
    // than two data-dependent loops keeps the trip count predictable.
    template <typename Int>
    inline int64_t merged_key(const Int *chosen, int size, Int feature, int64_t n_features)
    {
        int64_t key = 0;
        int taken = 0;
        bool placed = false;
        for (int i = 0; i < size + 1; ++i)
        {
            Int current;
            if (!placed && (taken == size || chosen[taken] > feature))
            {
                current = feature;
                placed = true;
            }
            else
            {
                current = chosen[taken++];
            }
            key = key * n_features + static_cast<int64_t>(current);
        }
        return key;
    }

    // The same splice, materialized: `out` receives the sorted tuple of size + 1 features.
    template <typename Int>
    inline void merged_row(const Int *chosen, int size, Int feature, int32_t *out)
    {
        int taken = 0;
        bool placed = false;
        for (int i = 0; i < size + 1; ++i)
        {
            if (!placed && (taken == size || chosen[taken] > feature))
            {
                out[i] = static_cast<int32_t>(feature);
                placed = true;
            }
            else
            {
                out[i] = static_cast<int32_t>(chosen[taken++]);
            }
        }
    }

    // Flat hash index of the tables. `starts` and `shifts` point into a Geometry.
    struct SubsetIndex
    {
        const int64_t *keys = nullptr;    // slot keys, -1 = empty
        const int32_t *rows = nullptr;    // slot row ids within the order's table
        const int64_t *starts = nullptr;  // per order: first slot of its block
        const int64_t *shifts = nullptr;  // per order: 64 - log2(block size)

        // Row of the order-s subset with the given key, or -1 when it is not in the table.
        // Bounded by the block size, so even a malformed index cannot loop forever.
        int64_t row(int64_t key, int s, const int64_t *counts) const
        {
            const int64_t block = starts[s];
            const int shift = static_cast<int>(shifts[s]);
            const uint64_t mask = (uint64_t(1) << (64 - shift)) - 1;
            uint64_t slot = (static_cast<uint64_t>(key) * kIndexHashMultiplier) >> shift;
            for (uint64_t probe = 0; probe <= mask; ++probe)
            {
                const int64_t stored = keys[block + slot];
                if (stored == key)
                {
                    const int64_t r = rows[block + slot];
                    return (r >= 0 && r < counts[s]) ? r : -1;
                }
                if (stored < 0)
                    return -1;
                slot = (slot + 1) & mask;
            }
            return -1;
        }
    };

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
            uint64_t slot = (static_cast<uint64_t>(key) * kIndexHashMultiplier) >> shift;
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

    // The tables of an ensemble in the kernels' flat layout, plus their index when the keys fit.
    struct Tables
    {
        std::vector<int32_t> keys;    // per order its sorted rows back to back, `order` ints each
        std::vector<int64_t> counts;  // per order: number of rows
        bool has_index = false;       // false when the key encoding overflows an int64
        std::vector<int64_t> slot_keys;
        std::vector<int32_t> slot_rows;
        bool too_many_rows = false;  // a table beyond INT32_MAX rows (row ids are int32)
    };

    // Structural DFS over every root-to-leaf path, collecting each feature subset (orders
    // 2..max_order) that co-occurs on some path: the only interactions an explanation can touch.
    // A subset is emitted at the first occurrence of its deepest member (a feature repeated on a
    // path adds nothing new), and both children are always walked. Subsets are kept as int64
    // keys while the encoding fits, else as sorted rows.
    class SubsetCollector
    {
    public:
        SubsetCollector(int64_t n_features, int max_order)
            : n_features_(n_features), max_order_(max_order),
              keys_fit_(keys_fit(n_features, max_order)),
              key_sets_(static_cast<size_t>(max_order) + 1),
              row_sets_(static_cast<size_t>(max_order) + 1),
              chosen_(static_cast<size_t>(std::max(max_order, 1))),
              merged_(static_cast<size_t>(std::max(max_order, 1)))
        {
        }

        // One tree; node ids are relative to the given arrays, and a leaf has
        // children_left == children_right (its feature entry is never read).
        template <typename Int>
        void walk(const Int *features, const Int *children_left, const Int *children_right, int64_t node)
        {
            if (children_left[node] == children_right[node])
                return;
            const int feature = static_cast<int>(features[node]);
            auto at = std::lower_bound(path_.begin(), path_.end(), feature);
            const bool first = (at == path_.end() || *at != feature);
            if (first)
            {
                emit(feature);
                path_.insert(at, feature);
            }
            walk(features, children_left, children_right, static_cast<int64_t>(children_left[node]));
            walk(features, children_left, children_right, static_cast<int64_t>(children_right[node]));
            if (first)
                path_.erase(std::lower_bound(path_.begin(), path_.end(), feature));
        }

        // The per-order tables in lexicographic (= numeric key) order and, when the keys fit,
        // the hash index mapping each key to its row (block geometry as derive_geometry).
        Tables finish() const
        {
            Tables t;
            t.counts.assign(static_cast<size_t>(max_order_) + 1, 0);
            t.has_index = keys_fit_;
            int64_t position = 0, slot_position = 0;
            for (int order = 2; order <= max_order_; ++order)
            {
                if (keys_fit_)
                {
                    const std::vector<int64_t> sorted = key_sets_[order].sorted_keys();
                    const int64_t count = static_cast<int64_t>(sorted.size());
                    if (count > INT32_MAX)
                    {
                        t.too_many_rows = true;
                        return t;
                    }
                    t.counts[order] = count;
                    t.keys.resize(static_cast<size_t>(position + count * order));
                    const int bits = block_bits(count);
                    const int64_t size = int64_t(1) << bits;
                    const uint64_t mask = static_cast<uint64_t>(size) - 1;
                    t.slot_keys.resize(static_cast<size_t>(slot_position + size), -1);
                    t.slot_rows.resize(static_cast<size_t>(slot_position + size), -1);
                    int64_t *block_keys = t.slot_keys.data() + slot_position;
                    int32_t *block_rows = t.slot_rows.data() + slot_position;
                    for (int64_t r = 0; r < count; ++r)
                    {
                        int64_t key = sorted[static_cast<size_t>(r)];
                        uint64_t slot = (static_cast<uint64_t>(key) * kIndexHashMultiplier) >> (64 - bits);
                        while (block_keys[slot] >= 0)
                            slot = (slot + 1) & mask;
                        block_keys[slot] = key;
                        block_rows[slot] = static_cast<int32_t>(r);
                        int32_t *row = t.keys.data() + position + r * order;
                        for (int d = order - 1; d >= 0; --d)
                        {
                            row[d] = static_cast<int32_t>(key % n_features_);
                            key /= n_features_;
                        }
                    }
                    slot_position += size;
                }
                else
                {
                    const auto &rows = row_sets_[order];  // a std::set iterates in sorted order
                    const int64_t count = static_cast<int64_t>(rows.size());
                    if (count > INT32_MAX)
                    {
                        t.too_many_rows = true;
                        return t;
                    }
                    t.counts[order] = count;
                    t.keys.reserve(static_cast<size_t>(position + count * order));
                    for (const auto &row : rows)
                        t.keys.insert(t.keys.end(), row.begin(), row.end());
                }
                position += t.counts[order] * order;
            }
            return t;
        }

    private:
        // every (order-1)-subset of the current path, merged with new_feature
        void emit(int new_feature)
        {
            for (int order = 2; order <= max_order_; ++order)
            {
                if (static_cast<int>(path_.size()) < order - 1)
                    break;
                choose(order - 1, 0, 0, new_feature, order);
            }
        }

        void choose(int remaining, int start, int level, int new_feature, int order)
        {
            if (remaining == 0)
            {
                if (keys_fit_)
                {
                    key_sets_[order].insert(merged_key(chosen_.data(), level, new_feature, n_features_));
                }
                else
                {
                    merged_row(chosen_.data(), level, new_feature, merged_.data());
                    row_sets_[order].insert(std::vector<int32_t>(merged_.begin(), merged_.begin() + order));
                }
                return;
            }
            const int n = static_cast<int>(path_.size());
            for (int i = start; i <= n - remaining; ++i)
            {
                chosen_[static_cast<size_t>(level)] = path_[static_cast<size_t>(i)];
                choose(remaining - 1, i + 1, level + 1, new_feature, order);
            }
        }

        int64_t n_features_;
        int max_order_;
        bool keys_fit_;
        std::vector<int> path_;  // sorted distinct features on the current path
        std::vector<FlatKeySet> key_sets_;                       // per order, while the keys fit
        std::vector<std::set<std::vector<int32_t>>> row_sets_;  // per order, otherwise
        std::vector<int> chosen_;
        std::vector<int32_t> merged_;
    };
}  // namespace subset_tables

#ifdef Py_PYTHON_H
// === preprocess_subset_tables (Python binding) ===
// preprocess_subset_tables(features, children_left, children_right, tree_offsets, n_features,
//                          max_order)
//   -> ((keys, counts), (slot_keys, slot_rows) or None)
// The node arrays are the ensemble's trees back to back with tree-relative child ids: tree t
// owns nodes [tree_offsets[t], tree_offsets[t + 1]). A leaf has children_left ==
// children_right (its feature entry is ignored); split features must lie in [0, n_features).
// The tables are always returned; the index is None when n_features^max_order overflows an
// int64, and the caller's kernel then searches the tables (quadrature) or takes another route
// (interventional). Everything else about the layout follows from `counts` (derive_geometry).
namespace subset_tables
{
    template <typename T>
    inline PyObject *vector_to_array(const std::vector<T> &values, int typenum)
    {
        npy_intp n = static_cast<npy_intp>(values.size());
        PyObject *array = PyArray_SimpleNew(1, &n, typenum);
        if (array && n > 0)
            std::memcpy(PyArray_DATA((PyArrayObject *)array), values.data(), values.size() * sizeof(T));
        return array;
    }

    // Checks the arrays a kernel receives against the geometry derived from `counts`; returns a
    // message or NULL. `n_slots` is the length of both index arrays (-1 when there is no index).
    inline const char *check_layout(const Geometry &g, int64_t n_keys, int64_t n_slots, int64_t n_outputs)
    {
        if (n_keys != g.n_keys)
            return "subset tables are inconsistent with the subset_keys length.";
        if (n_slots >= 0 && n_slots != g.n_slots)
            return "subset index arrays do not match the block sizes derived from counts.";
        if (n_outputs >= 0 && n_outputs != g.n_outputs)
            return "out must have one entry per interaction of the layout.";
        return NULL;
    }
}  // namespace subset_tables

static PyObject *preprocess_subset_tables(PyObject *self, PyObject *args)
{
    (void)self;
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

    subset_tables::SubsetCollector collector(n_features, max_order);
    for (int64_t t = 0; t < n_trees; ++t)
        collector.walk(features + toff[t], cl + toff[t], cr + toff[t], 0);
    release();
    const subset_tables::Tables tables = collector.finish();
    if (tables.too_many_rows)
    {
        PyErr_SetString(PyExc_ValueError, "subset tables beyond 2^31 rows per order are not supported.");
        return NULL;
    }

    PyObject *keys = subset_tables::vector_to_array(tables.keys, NPY_INT32);
    PyObject *counts = subset_tables::vector_to_array(tables.counts, NPY_INT64);
    PyObject *index = NULL;
    bool failed = !keys || !counts;
    if (!failed && tables.has_index)
    {
        PyObject *slot_keys = subset_tables::vector_to_array(tables.slot_keys, NPY_INT64);
        PyObject *slot_rows = subset_tables::vector_to_array(tables.slot_rows, NPY_INT32);
        if (!slot_keys || !slot_rows)
        {
            Py_XDECREF(slot_keys);
            Py_XDECREF(slot_rows);
            failed = true;
        }
        else
        {
            index = Py_BuildValue("(NN)", slot_keys, slot_rows);
            failed = index == NULL;
        }
    }
    else if (!failed)
    {
        Py_INCREF(Py_None);
        index = Py_None;
    }
    if (failed)
    {
        Py_XDECREF(keys);
        Py_XDECREF(counts);
        return NULL;
    }
    return Py_BuildValue("((NN)N)", keys, counts, index);
}
// === layout_to_dict (Python binding) ===
// layout_to_dict(out, keys, counts, n_features, min_order, feature_ids, skip_zeros)
//   -> {sorted feature tuple: value}
// Reads a kernel's output array (the layout of derive_geometry) back into a dict: the order-1
// block by feature id, then every table row's features. `feature_ids` maps the kernel's
// feature ids to the ids the result carries (int64 array of n_features entries), or None to
// keep them. With skip_zeros, entries that are exactly zero are left out. Feature ids are
// created once and shared between the tuples.
static PyObject *layout_to_dict(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *out_obj, *keys_obj, *counts_obj, *feature_ids_obj;
    int n_features, min_order, skip_zeros;
    if (!PyArg_ParseTuple(args, "OOOiiOp", &out_obj, &keys_obj, &counts_obj, &n_features, &min_order,
                          &feature_ids_obj, &skip_zeros))
        return NULL;
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_FROM_OTF(out_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *keys_arr = (PyArrayObject *)PyArray_FROM_OTF(keys_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *counts_arr = (PyArrayObject *)PyArray_FROM_OTF(counts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ids_arr = (feature_ids_obj != Py_None)
                                 ? (PyArrayObject *)PyArray_FROM_OTF(feature_ids_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY)
                                 : NULL;
    auto release = [&]()
    {
        Py_XDECREF(out_arr);
        Py_XDECREF(keys_arr);
        Py_XDECREF(counts_arr);
        Py_XDECREF(ids_arr);
    };
    const char *arg_error = NULL;
    if (!out_arr || !keys_arr || !counts_arr || (feature_ids_obj != Py_None && !ids_arr))
        arg_error = "out (float64), keys (int32), counts (int64) and feature_ids (int64 or None) must be numpy arrays.";
    else if (PyArray_NDIM(out_arr) != 1 || PyArray_NDIM(keys_arr) != 1 || PyArray_NDIM(counts_arr) != 1 ||
             (ids_arr && PyArray_NDIM(ids_arr) != 1))
        arg_error = "layout arrays must be 1-dimensional.";
    else if (n_features < 0 || min_order < 1 || PyArray_DIM(counts_arr, 0) < 1 ||
             (ids_arr && PyArray_DIM(ids_arr, 0) < n_features))
        arg_error = "n_features, min_order, counts or feature_ids are inconsistent.";
    if (arg_error)
    {
        release();
        PyErr_SetString(PyExc_ValueError, arg_error);
        return NULL;
    }
    const int max_order = static_cast<int>(PyArray_DIM(counts_arr, 0)) - 1;
    const int64_t *counts = (const int64_t *)PyArray_DATA(counts_arr);
    for (int order = 2; order <= max_order; ++order)
        if (counts[order] < 0)
            arg_error = "subset counts must be non-negative.";
    subset_tables::Geometry geometry;
    if (!arg_error)
    {
        geometry = subset_tables::derive_geometry(counts, max_order, n_features, min_order);
        arg_error = subset_tables::check_layout(geometry, (int64_t)PyArray_DIM(keys_arr, 0), -1,
                                                (int64_t)PyArray_DIM(out_arr, 0));
    }
    if (arg_error)
    {
        release();
        PyErr_SetString(PyExc_ValueError, arg_error);
        return NULL;
    }
    const double *out = (const double *)PyArray_DATA(out_arr);
    const int32_t *keys = (const int32_t *)PyArray_DATA(keys_arr);
    const int64_t *ids = ids_arr ? (const int64_t *)PyArray_DATA(ids_arr) : NULL;

    PyObject *output = PyDict_New();
    std::vector<PyObject *> id_objects(static_cast<size_t>(n_features), nullptr);
    bool failed = output == NULL;
    auto feature_id = [&](int32_t f) -> PyObject *
    {
        PyObject *&id = id_objects[static_cast<size_t>(f)];
        if (!id)
            id = PyLong_FromLongLong(ids ? static_cast<long long>(ids[f]) : static_cast<long long>(f));
        if (id)
            Py_INCREF(id);
        return id;  // new reference, or NULL
    };
    auto put = [&](PyObject *key, double value)
    {
        PyObject *item = key ? PyFloat_FromDouble(value) : NULL;
        if (!item || PyDict_SetItem(output, key, item) < 0)
            failed = true;
        Py_XDECREF(item);
        Py_XDECREF(key);
    };
    if (min_order <= 1)
    {
        for (int f = 0; f < n_features && !failed; ++f)
        {
            if (skip_zeros && out[f] == 0.0)
                continue;
            PyObject *key = PyTuple_New(1);
            if (key)
                PyTuple_SET_ITEM(key, 0, feature_id(static_cast<int32_t>(f)));
            put(key, out[f]);
        }
    }
    for (int order = std::max(min_order, 2); order <= max_order && !failed; ++order)
    {
        const int32_t *row = keys + geometry.table_starts[order];
        const double *values = out + geometry.output_offsets[order];
        for (int64_t r = 0; r < counts[order] && !failed; ++r, row += order)
        {
            if (skip_zeros && values[r] == 0.0)
                continue;
            PyObject *key = PyTuple_New(order);
            for (int i = 0; key && i < order; ++i)
                PyTuple_SET_ITEM(key, i, feature_id(row[i]));
            put(key, values[r]);
        }
    }
    for (PyObject *id : id_objects)
        Py_XDECREF(id);
    release();
    if (failed || PyErr_Occurred())
    {
        Py_XDECREF(output);
        return NULL;
    }
    return output;
}
#endif  // Py_PYTHON_H
