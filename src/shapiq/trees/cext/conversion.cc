// Booster-dump parsers: XGBoost UBJSON and LightGBM text.
//
// Conversion sits on the hot path of tree explanations, so the byte loops
// run in C: the XGBoost parser consumes save_raw() (UBJSON, the fast dump)
// and the LightGBM parser consumes model_to_string(). Both return flat
// per-forest arrays as bytes; every modelling policy — threshold
// semantics, base scores, multiclass layout — stays on the Python side,
// which also owns a pure-Python fallback over the slow dumps. Parsers
// raise ValueError on anything unexpected; the Python side falls back.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------- outputs

PyObject *bytes_from_int64(const std::vector<int64_t> &values) {
    return PyBytes_FromStringAndSize(reinterpret_cast<const char *>(values.data()),
                                     static_cast<Py_ssize_t>(values.size() * sizeof(int64_t)));
}

PyObject *bytes_from_double(const std::vector<double> &values) {
    return PyBytes_FromStringAndSize(reinterpret_cast<const char *>(values.data()),
                                     static_cast<Py_ssize_t>(values.size() * sizeof(double)));
}

struct ForestArrays {
    std::vector<int64_t> node_counts;
    std::vector<int64_t> left;
    std::vector<int64_t> right;
    std::vector<int64_t> features;
    std::vector<double> thresholds;  // xgboost: split_conditions incl. leaf values
    std::vector<double> values;      // lightgbm only; empty for xgboost
    std::vector<int64_t> tree_info;  // xgboost only; empty for lightgbm
    int64_t trees_per_iteration = 1;
};

// ------------------------------------------------------------- ubjson

class ByteStream {
public:
    ByteStream(const uint8_t *data, size_t size) : data_(data), size_(size) {}

    static uint32_t bswap32(uint32_t v) {
        return ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) | ((v & 0x00FF0000u) >> 8) |
               ((v & 0xFF000000u) >> 24);
    }
    static uint64_t bswap64(uint64_t v) {
        return ((v & 0x00000000000000FFull) << 56) | ((v & 0x000000000000FF00ull) << 40) |
               ((v & 0x0000000000FF0000ull) << 24) | ((v & 0x00000000FF000000ull) << 8) |
               ((v & 0x000000FF00000000ull) >> 8) | ((v & 0x0000FF0000000000ull) >> 24) |
               ((v & 0x00FF000000000000ull) >> 40) | ((v & 0xFF00000000000000ull) >> 56);
    }

    void require(size_t bytes) const {
        if (pos_ > size_ || bytes > size_ - pos_) {
            throw ParseError("unexpected end of UBJSON stream");
        }
    }
    uint8_t readByte() {
        require(1);
        return data_[pos_++];
    }
    int64_t readIntPayload(uint8_t marker) {
        switch (marker) {
            case 'i':
                return static_cast<int8_t>(readByte());
            case 'U':
                return readByte();
            case 'I': {
                require(2);
                uint16_t v = (static_cast<uint16_t>(data_[pos_]) << 8) | data_[pos_ + 1];
                pos_ += 2;
                return static_cast<int16_t>(v);
            }
            case 'l': {
                require(4);
                uint32_t v;
                std::memcpy(&v, data_ + pos_, 4);
                pos_ += 4;
                return static_cast<int32_t>(bswap32(v));
            }
            case 'L': {
                require(8);
                uint64_t v;
                std::memcpy(&v, data_ + pos_, 8);
                pos_ += 8;
                return static_cast<int64_t>(bswap64(v));
            }
            default:
                throw ParseError("invalid integer marker in UBJSON stream");
        }
    }
    double readFloatPayload(uint8_t marker) {
        if (marker == 'd') {
            require(4);
            uint32_t v;
            std::memcpy(&v, data_ + pos_, 4);
            pos_ += 4;
            v = bswap32(v);
            float f;
            std::memcpy(&f, &v, 4);
            return static_cast<double>(f);
        }
        if (marker == 'D') {
            require(8);
            uint64_t v;
            std::memcpy(&v, data_ + pos_, 8);
            pos_ += 8;
            v = bswap64(v);
            double d;
            std::memcpy(&d, &v, 8);
            return d;
        }
        return static_cast<double>(readIntPayload(marker));
    }
    static bool isIntMarker(uint8_t marker) {
        return marker == 'i' || marker == 'U' || marker == 'I' || marker == 'l' || marker == 'L';
    }

    // Scan forward for a length-prefixed object key (the XGBoost writer
    // prefixes every key with an 'L' int64 length; other integer markers are
    // accepted too). Callers sequence scans so only structural bytes lie
    // between the current position and the key; a miss raises and the
    // Python side falls back.
    void scanKey(size_t length, const char *key) {
        while (pos_ < size_) {
            uint8_t marker = data_[pos_];
            size_t payload = markerPayloadSize(marker);
            if (payload != 0 && pos_ + 1 + payload + length <= size_) {
                uint64_t declared = readBigEndian(data_ + pos_ + 1, payload);
                if (declared == length &&
                    std::memcmp(data_ + pos_ + 1 + payload, key, length) == 0) {
                    pos_ += 1 + payload + length;
                    return;
                }
            }
            pos_++;
        }
        throw ParseError(std::string("key not found in UBJSON stream: ") + key);
    }

    static size_t markerPayloadSize(uint8_t marker) {
        switch (marker) {
            case 'i':
            case 'U':
                return 1;
            case 'I':
                return 2;
            case 'l':
                return 4;
            case 'L':
                return 8;
            default:
                return 0;
        }
    }
    static uint64_t readBigEndian(const uint8_t *bytes, size_t width) {
        uint64_t value = 0;
        for (size_t i = 0; i < width; i++) {
            value = (value << 8) | bytes[i];
        }
        return value;
    }

    int64_t readInt() {
        uint8_t marker = readByte();
        if (marker == 'S') {
            // string-encoded integer (gbtree_model_param counts)
            std::string text = readString();
            return std::strtoll(text.c_str(), nullptr, 10);
        }
        return readIntPayload(marker);
    }
    std::string readString() {
        uint8_t marker = readByte();
        int64_t length = readIntPayload(marker);
        if (length < 0) {
            throw ParseError("negative string length in UBJSON stream");
        }
        require(static_cast<size_t>(length));
        std::string value(reinterpret_cast<const char *>(data_ + pos_),
                          static_cast<size_t>(length));
        pos_ += static_cast<size_t>(length);
        return value;
    }

    // Positioned at '[': read the '$type #count' header of a typed array.
    // Plain '[]' arrays report zero elements; other plain arrays are not
    // produced by the XGBoost writer for the fields the parser touches.
    struct ArrayHeader {
        uint8_t type;
        uint64_t count;
    };
    ArrayHeader readArrayHeader() {
        uint8_t opener = readByte();
        if (opener != '[') {
            throw ParseError("expected a UBJSON array");
        }
        require(1);
        if (data_[pos_] == ']') {
            pos_++;
            return {0, 0};
        }
        if (data_[pos_] == '$') {
            pos_++;
            uint8_t type = readByte();
            if (readByte() != '#') {
                throw ParseError("typed UBJSON array without a count");
            }
            uint8_t count_marker = readByte();
            int64_t count = readIntPayload(count_marker);
            if (count < 0) {
                throw ParseError("negative UBJSON array count");
            }
            return {type, static_cast<uint64_t>(count)};
        }
        if (data_[pos_] == '#') {
            pos_++;
            uint8_t count_marker = readByte();
            int64_t count = readIntPayload(count_marker);
            if (count < 0) {
                throw ParseError("negative UBJSON array count");
            }
            return {0, static_cast<uint64_t>(count)};
        }
        throw ParseError("untyped UBJSON array where a typed one was expected");
    }

    // untyped-but-counted arrays ('[#') carry a marker per element
    uint8_t elementMarker(const ArrayHeader &header) {
        return header.type != 0 ? header.type : readByte();
    }

    uint64_t readInt64Array(std::vector<int64_t> &out) {
        ArrayHeader header = readArrayHeader();
        out.reserve(out.size() + header.count);
        for (uint64_t i = 0; i < header.count; i++) {
            out.push_back(readIntPayload(elementMarker(header)));
        }
        return header.count;
    }
    uint64_t readFloat64Array(std::vector<double> &out) {
        ArrayHeader header = readArrayHeader();
        if (header.type == 'D') {
            require(header.count * 8);
            size_t start = out.size();
            out.resize(start + header.count);
            std::memcpy(out.data() + start, data_ + pos_, header.count * 8);
            pos_ += header.count * 8;
            for (uint64_t i = 0; i < header.count; i++) {
                uint64_t raw;
                std::memcpy(&raw, &out[start + i], 8);
                raw = bswap64(raw);
                std::memcpy(&out[start + i], &raw, 8);
            }
            return header.count;
        }
        out.reserve(out.size() + header.count);
        for (uint64_t i = 0; i < header.count; i++) {
            out.push_back(readFloatPayload(elementMarker(header)));
        }
        return header.count;
    }
    uint64_t skipArray() {
        ArrayHeader header = readArrayHeader();
        for (uint64_t i = 0; i < header.count; i++) {
            uint8_t marker = elementMarker(header);
            if (marker == 'd' || marker == 'D') {
                readFloatPayload(marker);
            } else {
                readIntPayload(marker);
            }
        }
        return header.count;
    }

private:
    const uint8_t *data_;
    size_t size_ = 0;
    size_t pos_ = 0;
};

ForestArrays parse_xgboost(const uint8_t *data, size_t size) {
    ByteStream stream(data, size);
    ForestArrays forest;
    stream.scanKey(9, "num_trees");
    int64_t num_trees = stream.readInt();
    if (num_trees < 0) {
        throw ParseError("negative tree count in the booster");
    }
    // iteration_indptr is an integer array payload; skipping it explicitly
    // keeps the following scans on structural bytes only
    stream.scanKey(16, "iteration_indptr");
    stream.skipArray();
    stream.scanKey(9, "tree_info");
    stream.readInt64Array(forest.tree_info);
    if (forest.tree_info.size() != static_cast<size_t>(num_trees)) {
        throw ParseError("tree_info length disagrees with the tree count");
    }
    stream.scanKey(5, "trees");
    for (int64_t tree = 0; tree < num_trees; tree++) {
        // fields arrive in the writer's fixed order; every array between two
        // reads is skipped explicitly so key scans never cross payload bytes
        stream.scanKey(12, "base_weights");
        uint64_t n_nodes = stream.skipArray();
        stream.scanKey(16, "categories_nodes");
        if (stream.skipArray() != 0) {
            throw ParseError(
                "the booster uses categorical splits, which the unified tree "
                "layout does not represent");
        }
        stream.scanKey(19, "categories_segments");
        stream.skipArray();
        stream.scanKey(16, "categories_sizes");
        stream.skipArray();
        stream.scanKey(12, "default_left");
        stream.skipArray();
        stream.scanKey(13, "left_children");
        uint64_t n_left = stream.readInt64Array(forest.left);
        stream.scanKey(12, "loss_changes");
        stream.skipArray();
        stream.scanKey(7, "parents");
        stream.skipArray();
        stream.scanKey(14, "right_children");
        uint64_t n_right = stream.readInt64Array(forest.right);
        stream.scanKey(16, "split_conditions");
        uint64_t n_conditions = stream.readFloat64Array(forest.thresholds);
        stream.scanKey(13, "split_indices");
        uint64_t n_features = stream.readInt64Array(forest.features);
        stream.scanKey(10, "split_type");
        stream.skipArray();
        stream.scanKey(11, "sum_hessian");
        stream.skipArray();
        if (n_left != n_nodes || n_right != n_nodes || n_conditions != n_nodes ||
            n_features != n_nodes) {
            throw ParseError("tree arrays disagree on the node count");
        }
        forest.node_counts.push_back(static_cast<int64_t>(n_nodes));
    }
    return forest;
}

// ------------------------------------------------------------- lightgbm

class TextStream {
public:
    TextStream(const char *data, size_t size) : data_(data), size_(size) {}

    bool scanKey(const char *key) {
        size_t length = std::strlen(key);
        while (pos_ + length <= size_) {
            // keys sit at the start of a line
            if ((pos_ == 0 || data_[pos_ - 1] == '\n') &&
                std::memcmp(data_ + pos_, key, length) == 0) {
                pos_ += length;
                return true;
            }
            pos_++;
        }
        return false;
    }
    void requireKey(const char *key) {
        if (!scanKey(key)) {
            throw ParseError(std::string("key not found in LightGBM dump: ") + key);
        }
    }
    int64_t readInt() {
        char *end = nullptr;
        int64_t value = std::strtoll(data_ + pos_, &end, 10);
        if (end == data_ + pos_) {
            throw ParseError("expected an integer in the LightGBM dump");
        }
        pos_ = static_cast<size_t>(end - data_);
        return value;
    }
    void readIntLine(std::vector<int64_t> &out) {
        while (pos_ < size_ && data_[pos_] != '\n') {
            while (pos_ < size_ && (data_[pos_] == ' ' || data_[pos_] == '\t')) {
                pos_++;
            }
            if (pos_ >= size_ || data_[pos_] == '\n') {
                break;
            }
            out.push_back(readInt());
        }
    }
    void readDoubleLine(std::vector<double> &out) {
        while (pos_ < size_ && data_[pos_] != '\n') {
            while (pos_ < size_ && (data_[pos_] == ' ' || data_[pos_] == '\t')) {
                pos_++;
            }
            if (pos_ >= size_ || data_[pos_] == '\n') {
                break;
            }
            char *end = nullptr;
            double value = std::strtod(data_ + pos_, &end);
            if (end == data_ + pos_) {
                throw ParseError("expected a number in the LightGBM dump");
            }
            out.push_back(value);
            pos_ = static_cast<size_t>(end - data_);
        }
    }

private:
    const char *data_;
    size_t size_ = 0;
    size_t pos_ = 0;
};

int64_t remap_child(int64_t child, int64_t n_internal) {
    // internal children keep their id; leaves are encoded as ~id
    return child >= 0 ? child : n_internal - child - 1;
}

ForestArrays parse_lightgbm(const char *data, size_t size) {
    TextStream stream(data, size);
    ForestArrays forest;
    {
        TextStream header = stream;
        if (header.scanKey("num_tree_per_iteration=")) {
            forest.trees_per_iteration = header.readInt();
        }
    }
    while (stream.scanKey("Tree=")) {
        stream.readInt();
        stream.requireKey("num_leaves=");
        int64_t n_leaves = stream.readInt();
        if (n_leaves < 1) {
            throw ParseError("a LightGBM tree reports no leaves");
        }
        int64_t n_internal = n_leaves - 1;
        int64_t n_nodes = n_internal + n_leaves;
        std::vector<int64_t> features, decisions, lefts, rights;
        std::vector<double> thresholds, leaf_values;
        stream.requireKey("split_feature=");
        stream.readIntLine(features);
        stream.requireKey("threshold=");
        stream.readDoubleLine(thresholds);
        stream.requireKey("decision_type=");
        stream.readIntLine(decisions);
        stream.requireKey("left_child=");
        stream.readIntLine(lefts);
        stream.requireKey("right_child=");
        stream.readIntLine(rights);
        stream.requireKey("leaf_value=");
        stream.readDoubleLine(leaf_values);
        if (features.size() < static_cast<size_t>(n_internal) ||
            thresholds.size() < static_cast<size_t>(n_internal) ||
            decisions.size() < static_cast<size_t>(n_internal) ||
            lefts.size() < static_cast<size_t>(n_internal) ||
            rights.size() < static_cast<size_t>(n_internal) ||
            leaf_values.size() < static_cast<size_t>(n_leaves)) {
            throw ParseError("LightGBM tree arrays disagree with num_leaves");
        }
        for (int64_t node = 0; node < n_internal; node++) {
            if ((decisions[static_cast<size_t>(node)] & 1) != 0) {
                throw ParseError(
                    "the booster uses categorical splits, which the unified "
                    "tree layout does not represent");
            }
            forest.left.push_back(remap_child(lefts[static_cast<size_t>(node)], n_internal));
            forest.right.push_back(remap_child(rights[static_cast<size_t>(node)], n_internal));
            forest.features.push_back(features[static_cast<size_t>(node)]);
            forest.thresholds.push_back(thresholds[static_cast<size_t>(node)]);
            forest.values.push_back(0.0);
        }
        for (int64_t leaf = 0; leaf < n_leaves; leaf++) {
            forest.left.push_back(-1);
            forest.right.push_back(-1);
            forest.features.push_back(-2);
            forest.thresholds.push_back(0.0);
            forest.values.push_back(leaf_values[static_cast<size_t>(leaf)]);
        }
        forest.node_counts.push_back(n_nodes);
    }
    if (forest.node_counts.empty()) {
        throw ParseError("no trees found in the LightGBM dump");
    }
    return forest;
}

// ------------------------------------------------------------- module

struct Buffer {
    Py_buffer view{};
    bool held = false;

    ~Buffer() {
        if (held) {
            PyBuffer_Release(&view);
        }
    }
    bool acquire(PyObject *object) {
        if (PyObject_GetBuffer(object, &view, PyBUF_CONTIG_RO) != 0) {
            return false;
        }
        held = true;
        return true;
    }
};

PyObject *forest_to_tuple(const ForestArrays &forest, bool with_values) {
    PyObject *counts = bytes_from_int64(forest.node_counts);
    PyObject *left = bytes_from_int64(forest.left);
    PyObject *right = bytes_from_int64(forest.right);
    PyObject *features = bytes_from_int64(forest.features);
    PyObject *thresholds = bytes_from_double(forest.thresholds);
    PyObject *tail = with_values ? bytes_from_double(forest.values)
                                 : bytes_from_int64(forest.tree_info);
    if (!counts || !left || !right || !features || !thresholds || !tail) {
        Py_XDECREF(counts);
        Py_XDECREF(left);
        Py_XDECREF(right);
        Py_XDECREF(features);
        Py_XDECREF(thresholds);
        Py_XDECREF(tail);
        return nullptr;
    }
    PyObject *result = Py_BuildValue(
        "(NNNNNNl)", counts, left, right, features, thresholds, tail,
        static_cast<long>(forest.trees_per_iteration));
    return result;
}

PyObject *parse_xgboost_ubjson(PyObject *, PyObject *args) {
    PyObject *raw;
    if (!PyArg_ParseTuple(args, "O", &raw)) {
        return nullptr;
    }
    Buffer buffer;
    if (!buffer.acquire(raw)) {
        return nullptr;
    }
    try {
        ForestArrays forest = parse_xgboost(
            static_cast<const uint8_t *>(buffer.view.buf),
            static_cast<size_t>(buffer.view.len));
        return forest_to_tuple(forest, false);
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject *parse_lightgbm_text(PyObject *, PyObject *args) {
    PyObject *raw;
    if (!PyArg_ParseTuple(args, "O", &raw)) {
        return nullptr;
    }
    Buffer buffer;
    if (!buffer.acquire(raw)) {
        return nullptr;
    }
    try {
        ForestArrays forest = parse_lightgbm(
            static_cast<const char *>(buffer.view.buf),
            static_cast<size_t>(buffer.view.len));
        return forest_to_tuple(forest, true);
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"parse_xgboost_ubjson", parse_xgboost_ubjson, METH_VARARGS,
     "Parse an XGBoost save_raw() UBJSON dump into flat forest arrays."},
    {"parse_lightgbm_text", parse_lightgbm_text, METH_VARARGS,
     "Parse a LightGBM model_to_string() dump into flat forest arrays."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_conversion_cext",
    "Hot loops of the booster-dump conversions.", -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__conversion_cext(void) { return PyModule_Create(&module); }
