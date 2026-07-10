// Interventional tree accumulation kernel.
//
// The Python side extracts per-leaf present/absent constraints from the game
// and computes one coefficient per (|present|, |absent|, taken-from-present,
// taken-from-absent) shape from the index's declared discrete-derivative
// weights. This kernel only runs the hot loop: for every leaf it enumerates
// the interactions inside the leaf's constraint sets and accumulates
// value * coefficient into a sparse map keyed by the interaction's players.
//
// Interactions are packed into one uint64 as up to four 16-bit fields
// holding player id + 1 in ascending order (0 marks an unused slot), so the
// kernel serves orders up to four and up to 65534 players; the Python side
// falls back to the pure path beyond that.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

struct Buffer {
    Py_buffer view{};
    bool held = false;

    ~Buffer() {
        if (held) {
            PyBuffer_Release(&view);
        }
    }

    bool acquire(PyObject *object, const char *name) {
        if (PyObject_GetBuffer(object, &view, PyBUF_CONTIG_RO) != 0) {
            PyErr_Format(PyExc_TypeError, "%s must expose a contiguous buffer", name);
            return false;
        }
        held = true;
        return true;
    }

    const int64_t *as_int64() const { return static_cast<const int64_t *>(view.buf); }
    const double *as_double() const { return static_cast<const double *>(view.buf); }
    Py_ssize_t n_int64() const { return view.len / static_cast<Py_ssize_t>(sizeof(int64_t)); }
};

inline uint64_t pack_interaction(const int64_t *players, int count) {
    uint64_t key = 0;
    for (int slot = 0; slot < count; ++slot) {
        key |= (static_cast<uint64_t>(players[slot]) + 1) << (16 * slot);
    }
    return key;
}

// Enumerate all ways to take `take_present` members of `present` and
// `take_absent` members of `absent` (both ascending), merge each pick into
// ascending player order, and accumulate `contribution` on the packed key.
void accumulate_combinations(const int64_t *present, int n_present, int take_present,
                             const int64_t *absent, int n_absent, int take_absent,
                             double contribution,
                             std::unordered_map<uint64_t, double> &totals) {
    std::vector<int> pick_present(static_cast<size_t>(take_present));
    std::vector<int> pick_absent(static_cast<size_t>(take_absent));
    for (int i = 0; i < take_present; ++i) pick_present[static_cast<size_t>(i)] = i;
    int64_t merged[4];

    while (true) {
        for (int i = 0; i < take_absent; ++i) pick_absent[static_cast<size_t>(i)] = i;
        while (true) {
            // merge the two ascending picks into ascending player order
            int a = 0, b = 0, out = 0;
            while (a < take_present && b < take_absent) {
                const int64_t left = present[pick_present[static_cast<size_t>(a)]];
                const int64_t right = absent[pick_absent[static_cast<size_t>(b)]];
                merged[out++] = left < right ? (a++, left) : (b++, right);
            }
            while (a < take_present) merged[out++] = present[pick_present[static_cast<size_t>(a++)]];
            while (b < take_absent) merged[out++] = absent[pick_absent[static_cast<size_t>(b++)]];
            totals[pack_interaction(merged, out)] += contribution;

            // odometer over the absent pick
            int digit = take_absent - 1;
            while (digit >= 0 && pick_absent[static_cast<size_t>(digit)] == digit + n_absent - take_absent) --digit;
            if (digit < 0) break;
            ++pick_absent[static_cast<size_t>(digit)];
            for (int next = digit + 1; next < take_absent; ++next) {
                pick_absent[static_cast<size_t>(next)] = pick_absent[static_cast<size_t>(next - 1)] + 1;
            }
        }
        // odometer over the present pick
        int digit = take_present - 1;
        while (digit >= 0 && pick_present[static_cast<size_t>(digit)] == digit + n_present - take_present) --digit;
        if (digit < 0) break;
        ++pick_present[static_cast<size_t>(digit)];
        for (int next = digit + 1; next < take_present; ++next) {
            pick_present[static_cast<size_t>(next)] = pick_present[static_cast<size_t>(next - 1)] + 1;
        }
    }
}

PyObject *accumulate(PyObject *, PyObject *args) {
    PyObject *present_offsets_obj, *present_members_obj;
    PyObject *absent_offsets_obj, *absent_members_obj;
    PyObject *values_obj, *coefficients_obj;
    int max_present, max_absent, min_size, order;
    if (!PyArg_ParseTuple(args, "OOOOOOiiii", &present_offsets_obj, &present_members_obj,
                          &absent_offsets_obj, &absent_members_obj, &values_obj,
                          &coefficients_obj, &max_present, &max_absent, &min_size, &order)) {
        return nullptr;
    }
    Buffer present_offsets, present_members, absent_offsets, absent_members, values, coefficients;
    if (!present_offsets.acquire(present_offsets_obj, "present_offsets") ||
        !present_members.acquire(present_members_obj, "present_members") ||
        !absent_offsets.acquire(absent_offsets_obj, "absent_offsets") ||
        !absent_members.acquire(absent_members_obj, "absent_members") ||
        !values.acquire(values_obj, "values") || !coefficients.acquire(coefficients_obj, "coefficients")) {
        return nullptr;
    }
    if (order > 4) {
        PyErr_SetString(PyExc_ValueError, "the kernel packs at most four players per interaction");
        return nullptr;
    }
    const Py_ssize_t n_leaves = present_offsets.n_int64() - 1;
    if (n_leaves < 0 || absent_offsets.n_int64() != n_leaves + 1 ||
        values.view.len != n_leaves * static_cast<Py_ssize_t>(sizeof(double))) {
        PyErr_SetString(PyExc_ValueError, "leaf offset and value buffers disagree on the leaf count");
        return nullptr;
    }
    const int64_t *p_offsets = present_offsets.as_int64();
    const int64_t *p_members = present_members.as_int64();
    const int64_t *a_offsets = absent_offsets.as_int64();
    const int64_t *a_members = absent_members.as_int64();
    const double *leaf_values = values.as_double();
    const double *table = coefficients.as_double();
    // cross-buffer invariants: one inconsistent buffer would otherwise read
    // out of bounds, so every cheap invariant is validated, not trusted
    if (n_leaves > 0 && (p_offsets[0] != 0 || a_offsets[0] != 0)) {
        PyErr_SetString(PyExc_ValueError, "leaf offsets must start at zero");
        return nullptr;
    }
    for (Py_ssize_t leaf = 0; leaf < n_leaves; ++leaf) {
        const int64_t p_count = p_offsets[leaf + 1] - p_offsets[leaf];
        const int64_t a_count = a_offsets[leaf + 1] - a_offsets[leaf];
        if (p_count < 0 || a_count < 0) {
            PyErr_SetString(PyExc_ValueError, "leaf offsets must be non-decreasing");
            return nullptr;
        }
        if (p_count > max_present || a_count > max_absent) {
            PyErr_SetString(PyExc_ValueError,
                            "a leaf exceeds the declared coefficient-table extents");
            return nullptr;
        }
    }
    if (n_leaves > 0 && (p_offsets[n_leaves] > present_members.n_int64() ||
                         a_offsets[n_leaves] > absent_members.n_int64())) {
        PyErr_SetString(PyExc_ValueError, "member buffers are shorter than the offsets claim");
        return nullptr;
    }
    for (Py_ssize_t index = 0; n_leaves > 0 && index < p_offsets[n_leaves]; ++index) {
        if (p_members[index] < 0 || p_members[index] > 0xFFFD) {
            PyErr_SetString(PyExc_ValueError, "player ids must fit the 16-bit packing");
            return nullptr;
        }
    }
    for (Py_ssize_t index = 0; n_leaves > 0 && index < a_offsets[n_leaves]; ++index) {
        if (a_members[index] < 0 || a_members[index] > 0xFFFD) {
            PyErr_SetString(PyExc_ValueError, "player ids must fit the 16-bit packing");
            return nullptr;
        }
    }
    // coefficient layout: [e][r][a][b] with strides over (order + 1) sized last axes
    const int stride_b = 1;
    const int stride_a = (order + 1) * stride_b;
    const int stride_r = (order + 1) * stride_a;
    const int stride_e = (max_absent + 1) * stride_r;
    const Py_ssize_t expected = static_cast<Py_ssize_t>(max_present + 1) * stride_e *
                                static_cast<Py_ssize_t>(sizeof(double));
    if (coefficients.view.len != expected) {
        PyErr_SetString(PyExc_ValueError, "coefficient table does not match the declared extents");
        return nullptr;
    }

    std::unordered_map<uint64_t, double> totals;
    for (Py_ssize_t leaf = 0; leaf < n_leaves; ++leaf) {
        const int64_t *present = p_members + p_offsets[leaf];
        const int64_t *absent = a_members + a_offsets[leaf];
        const int n_present = static_cast<int>(p_offsets[leaf + 1] - p_offsets[leaf]);
        const int n_absent = static_cast<int>(a_offsets[leaf + 1] - a_offsets[leaf]);
        const double value = leaf_values[leaf];
        const int max_size = n_present + n_absent < order ? n_present + n_absent : order;
        for (int size = min_size; size <= max_size; ++size) {
            const int lowest = size - n_absent > 0 ? size - n_absent : 0;
            const int highest = size < n_present ? size : n_present;
            for (int take_present = lowest; take_present <= highest; ++take_present) {
                const int take_absent = size - take_present;
                const double coefficient =
                    table[n_present * stride_e + n_absent * stride_r + take_present * stride_a +
                          take_absent * stride_b];
                if (coefficient == 0.0) {
                    continue;
                }
                accumulate_combinations(present, n_present, take_present, absent, n_absent,
                                        take_absent, value * coefficient, totals);
            }
        }
    }

    PyObject *keys = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(totals.size() * sizeof(uint64_t)));
    PyObject *sums = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(totals.size() * sizeof(double)));
    if (keys == nullptr || sums == nullptr) {
        Py_XDECREF(keys);
        Py_XDECREF(sums);
        return nullptr;
    }
    auto *key_out = reinterpret_cast<uint64_t *>(PyBytes_AS_STRING(keys));
    auto *sum_out = reinterpret_cast<double *>(PyBytes_AS_STRING(sums));
    size_t cursor = 0;
    for (const auto &entry : totals) {
        key_out[cursor] = entry.first;
        sum_out[cursor] = entry.second;
        ++cursor;
    }
    PyObject *result = PyTuple_Pack(2, keys, sums);
    Py_DECREF(keys);
    Py_DECREF(sums);
    return result;
}

PyMethodDef methods[] = {
    {"accumulate", accumulate, METH_VARARGS,
     "Accumulate per-leaf interaction contributions into a sparse map."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_interventional_cext",
    "Hot loop of the interventional tree explainer.", -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__interventional_cext(void) { return PyModule_Create(&module); }
