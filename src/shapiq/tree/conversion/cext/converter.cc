/** This file contains the UBJSON parser for gradient boosting models. */
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <string>
#include <cstdlib>
#include <vector>

struct ParsedTreeArrays
{
    std::vector<int64_t> node_ids;
    std::vector<int64_t> feature_ids;
    std::vector<double> thresholds;
    std::vector<double> values;
    std::vector<int64_t> left_children;
    std::vector<int64_t> right_children;
    std::vector<int64_t> default_children;
    std::vector<double> node_sample_weights;
};

struct ParsedForest
{
    std::vector<ParsedTreeArrays> trees;
    int64_t num_class = 1;
    double base_score = 0.0;
};

class ByteStream
{
    // We assume big-endian encoding for multi-byte integers, as per UBJSON specification.
private:
    const uint8_t *data;
    size_t pos;
    size_t size;

public:
    ByteStream(const uint8_t *data, size_t size) : data(data), pos(0), size(size) {}
    bool isIntMarker(uint8_t marker) const
    {
        return marker == 'i' || marker == 'U' || marker == 'I' || marker == 'l' || marker == 'L';
    }

    // Byte-swap helpers for converting UBJSON big-endian integers to host order.
    // UBJSON is always big-endian; x86 and Apple-silicon macOS are little-endian.
    static uint32_t bswap32(uint32_t v) noexcept
    {
        return ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) | ((v & 0x00FF0000u) >> 8) | ((v & 0xFF000000u) >> 24);
    }
    static uint64_t bswap64(uint64_t v) noexcept
    {
        return ((v & 0x00000000000000FFull) << 56) | ((v & 0x000000000000FF00ull) << 40) | ((v & 0x0000000000FF0000ull) << 24) | ((v & 0x00000000FF000000ull) << 8) | ((v & 0x000000FF00000000ull) >> 8) | ((v & 0x0000FF0000000000ull) >> 24) | ((v & 0x00FF000000000000ull) >> 40) | ((v & 0xFF00000000000000ull) >> 56);
    }
    bool tryReadNonNegativeIntByMarker(uint8_t marker, uint64_t &value)
    {
        if (marker == 'i')
        {
            int8_t parsed = readInt8();
            if (parsed < 0)
            {
                return false;
            }
            value = static_cast<uint64_t>(parsed);
            return true;
        }
        if (marker == 'U')
        {
            value = static_cast<uint64_t>(readByte());
            return true;
        }
        if (marker == 'I')
        {
            int16_t parsed = readInt16();
            if (parsed < 0)
            {
                return false;
            }
            value = static_cast<uint64_t>(parsed);
            return true;
        }
        if (marker == 'l')
        {
            int32_t parsed = readInt32();
            if (parsed < 0)
            {
                return false;
            }
            value = static_cast<uint64_t>(parsed);
            return true;
        }
        if (marker == 'L')
        {
            int64_t parsed = readInt64();
            if (parsed < 0)
            {
                return false;
            }
            value = static_cast<uint64_t>(parsed);
            return true;
        }
        return false;
    }
    uint64_t readNonNegativeIntByMarker(uint8_t marker)
    {
        uint64_t value = 0;
        if (!tryReadNonNegativeIntByMarker(marker, value))
        {
            throw std::runtime_error("Invalid or negative integer value for marker: " + std::to_string(marker));
        }
        return value;
    }
    int8_t readInt8() { return static_cast<int8_t>(readByte()); }
    uint8_t readByte()
    {
        if (pos >= size)
            throw std::runtime_error("End of stream");
        return data[pos++];
    }
    int16_t readInt16()
    {
        uint16_t v = static_cast<uint16_t>(data[pos] << 8) | data[pos + 1];
        pos += 2;
        return static_cast<int16_t>(v);
    }
    int32_t readInt32()
    {
        uint32_t v;
        std::memcpy(&v, data + pos, 4);
        pos += 4;
        return static_cast<int32_t>(bswap32(v));
    }
    int64_t readInt64()
    {
        uint64_t v;
        std::memcpy(&v, data + pos, 8);
        pos += 8;
        return static_cast<int64_t>(bswap64(v));
    }
    float readFloat()
    {
        uint32_t v;
        std::memcpy(&v, data + pos, 4);
        pos += 4;
        v = bswap32(v);
        float f;
        std::memcpy(&f, &v, 4);
        return f;
    }
    double readDouble()
    {
        uint64_t v;
        std::memcpy(&v, data + pos, 8);
        pos += 8;
        v = bswap64(v);
        double d;
        std::memcpy(&d, &v, 8);
        return d;
    }

    std::string readString()
    {
        uint8_t marker = readByte();
        if (!isIntMarker(marker))
            throw std::runtime_error("Invalid marker for string length: " + std::to_string(marker));
        uint64_t length = readNonNegativeIntByMarker(marker);
        if (pos + length > size)
            throw std::runtime_error("End of stream while reading string");
        std::string value(reinterpret_cast<const char *>(data + pos), length);
        pos += length;
        return value;
    }

    int64_t readInt()
    {
        uint8_t marker = readByte();
        if (marker == 'i')
        { // int8
            return readInt8();
        }
        else if (marker == 'U')
        { // uint8
            return readByte();
        }
        else if (marker == 'I')
        { // int16
            return readInt16();
        }
        else if (marker == 'l')
        { // int32
            return readInt32();
        }
        else if (marker == 'L')
        { // int64
            return readInt64();
        }
        else if (marker == 'S')
        { // string-encoded integer (used by XGBoost for some params)
            std::string value = readString();
            return std::stoll(value);
        }
        else
        {
            throw std::runtime_error("Invalid marker for non-negative integer: " + std::to_string(marker));
        }
    }

    double readDoubleLike()
    {
        uint8_t marker = readByte();
        if (marker == 'd')
        {
            return static_cast<double>(readFloat());
        }
        if (marker == 'D')
        {
            return readDouble();
        }
        if (marker == 'i')
        {
            return static_cast<double>(readInt8());
        }
        if (marker == 'U')
        {
            return static_cast<double>(readByte());
        }
        if (marker == 'I')
        {
            return static_cast<double>(readInt16());
        }
        if (marker == 'l')
        {
            return static_cast<double>(readInt32());
        }
        if (marker == 'L')
        {
            return static_cast<double>(readInt64());
        }
        if (marker == 'S')
        {
            std::string value = readString();
            if (!value.empty() && value.front() == '[' && value.back() == ']')
            {
                value = value.substr(1, value.size() - 2);
            }
            return std::strtod(value.c_str(), nullptr);
        }
        if (marker == 'H')
        {
            std::string num_str = readString();
            return std::strtod(num_str.c_str(), nullptr);
        }
        throw std::runtime_error("Invalid marker for floating-point value: " + std::to_string(marker));
    }

    double readBaseScoreOrZero(int class_label = 0)
    {
        ByteStream s = *this;
        while (s.pos < s.size)
        {
            uint8_t marker = s.readByte();
            if (!s.isIntMarker(marker))
                continue;
            uint64_t length = 0;
            if (!s.tryReadNonNegativeIntByMarker(marker, length))
                continue;
            if (length != 10 || s.pos + 10 > s.size)
                continue;
            if (std::memcmp(s.data + s.pos, "base_score", 10) == 0)
            {
                s.pos += 10;
                // XGBoost 3 multi-class: base_score is a string-encoded CSV array
                // e.g. S L<len> "[-9.950876E-3,1.990199E-2,-9.950876E-3]"
                if (s.pos < s.size && s.data[s.pos] == 'S')
                {
                    s.pos++; // consume 'S'
                    std::string value = s.readString();
                    // Strip surrounding '[' ... ']' if present.
                    if (!value.empty() && value.front() == '[' && value.back() == ']')
                        value = value.substr(1, value.size() - 2);
                    // Parse the class_label-th comma-separated token.
                    int idx = 0;
                    size_t start = 0;
                    while (idx < class_label)
                    {
                        size_t comma = value.find(',', start);
                        if (comma == std::string::npos)
                            break; // fewer tokens than expected; use last one
                        start = comma + 1;
                        idx++;
                    }
                    return std::strtod(value.c_str() + start, nullptr);
                }
                return s.readDoubleLike();
            }
        }
        return 0.0;
    }

    int64_t readNumClassOrOne()
    {
        ByteStream s = *this;
        while (s.pos < s.size)
        {
            uint8_t marker = s.readByte();
            if (!s.isIntMarker(marker))
                continue;
            uint64_t length = 0;
            if (!s.tryReadNonNegativeIntByMarker(marker, length))
                continue;
            if (length != 9 || s.pos + 9 > s.size)
                continue;
            if (std::memcmp(s.data + s.pos, "num_class", 9) == 0)
            {
                s.pos += 9;
                int64_t val = s.readInt();
                return val;
            }
        }
        return 1;
    }

    int64_t readNumTrees()
    {
        while (pos < size)
        {
            uint8_t marker = readByte();
            if (!isIntMarker(marker))
                continue;
            uint64_t length = 0;
            if (!tryReadNonNegativeIntByMarker(marker, length))
                continue;
            if (length != 9 || pos + 9 > size)
                continue;
            if (std::memcmp(data + pos, "num_trees", 9) == 0)
            {
                pos += 9;
                return readInt();
            }
        }
        throw std::runtime_error("num_trees key not found");
    }

    // Reads past a UBJSON array opening '[' and any optional '$type#count' or
    // '#count' length prefix, leaving the stream positioned at the first element.
    // Without this, the trees-array header "[#L<count>" would be mistaken for a
    // 12-byte key by the skipTo scanner (the 'L' marker + int64 count = 12).
    // If the byte at pos is NOT '[', it is left unconsumed.
    void skipArrayHeader()
    {
        if (pos >= size || data[pos] != '[')
        {
            return; // not a '[', do not consume anything
        }
        pos++; // consume '['
        if (pos >= size)
            return;
        uint8_t next = data[pos];
        if (next == '$')
        {
            pos++; // consume '$'
            if (pos < size)
                pos++; // element type (skip)
            if (pos < size && data[pos] == '#')
            {
                pos++; // consume '#'
                if (pos < size)
                {
                    uint8_t cm = data[pos++];
                    uint64_t dummy = 0;
                    tryReadNonNegativeIntByMarker(cm, dummy); // consume count value bytes
                }
            }
        }
        else if (next == '#')
        {
            pos++; // consume '#'
            if (pos < size)
            {
                uint8_t cm = data[pos++];
                uint64_t dummy = 0;
                tryReadNonNegativeIntByMarker(cm, dummy); // consume count value bytes
            }
        }
        // else: plain '[' followed by first-element marker -- '[' already consumed, next byte left in place
    }
    void skipToTrees()
    {
        while (pos < size)
        {
            uint8_t marker = readByte();
            if (!isIntMarker(marker))
                continue;
            uint64_t length = 0;
            if (!tryReadNonNegativeIntByMarker(marker, length))
                continue;
            if (length != 5 || pos + 5 > size)
                continue;
            if (std::memcmp(data + pos, "trees", 5) == 0)
            {
                pos += 5;
                return;
            }
        }
        throw std::runtime_error("trees key not found");
    }

    // Reads just the element-count from a typed UBJSON array header ([$T#count...]).
    // The stream must be positioned right before the opening '[' or '$' marker.
    // After the call the stream is positioned at the first data element.
    uint64_t readArrayCount()
    {
        uint8_t first_marker = readNextValueMarker(); // skips '[', returns next non-bracket
        if (first_marker == '$')
        {
            /* uint8_t type_marker = */ readByte(); // skip element type (e.g. 'D' or 'l')
            uint8_t cp = readByte();
            if (cp != '#')
                throw std::runtime_error("Expected '#' after '$' in typed UBJSON array.");
            uint8_t count_marker = readByte();
            return readNonNegativeIntByMarker(count_marker);
        }
        if (first_marker == '#')
        {
            uint8_t count_marker = readByte();
            return readNonNegativeIntByMarker(count_marker);
        }
        throw std::runtime_error("readArrayCount: unexpected leading marker: " + std::to_string(first_marker));
    }

    uint8_t readNextValueMarker()
    {
        while (true)
        {
            uint8_t marker = readByte();
            if (marker == '[' || marker == ']')
            {
                continue;
            }
            return marker;
        }
    }

    double readTypedValueAsDouble(uint8_t type_marker)
    {
        if (type_marker == 'd')
        {
            return static_cast<double>(readFloat());
        }
        if (type_marker == 'D')
        {
            return readDouble();
        }
        if (type_marker == 'i')
        {
            return static_cast<double>(readInt8());
        }
        if (type_marker == 'U')
        {
            return static_cast<double>(readByte());
        }
        if (type_marker == 'I')
        {
            return static_cast<double>(readInt16());
        }
        if (type_marker == 'l')
        {
            return static_cast<double>(readInt32());
        }
        if (type_marker == 'L')
        {
            return static_cast<double>(readInt64());
        }
        throw std::runtime_error("Unsupported typed marker for value: " + std::to_string(type_marker));
    }

    int64_t readTypedValueAsInt64(uint8_t type_marker)
    {
        if (type_marker == 'i')
        {
            return static_cast<int64_t>(readInt8());
        }
        if (type_marker == 'U')
        {
            return static_cast<int64_t>(readByte());
        }
        if (type_marker == 'I')
        {
            return static_cast<int64_t>(readInt16());
        }
        if (type_marker == 'l')
        {
            return static_cast<int64_t>(readInt32());
        }
        if (type_marker == 'L')
        {
            return static_cast<int64_t>(readInt64());
        }
        throw std::runtime_error("Unsupported typed marker for integer value: " + std::to_string(type_marker));
    }

    void fillArray(double *array, uint64_t num_nodes, double const_value=0.0)
    {
        // This method fills the values_array with the values from the tree structure.
        uint8_t first_marker = readNextValueMarker();

        if (first_marker == '$')
        {
            uint8_t type_marker = readByte();
            uint8_t count_prefix = readByte();
            if (count_prefix != '#')
            {
                throw std::runtime_error("Expected '#' after UBJSON typed container marker '$'.");
            }
            uint8_t count_marker = readByte();
            uint64_t declared_count = readNonNegativeIntByMarker(count_marker);
            if (declared_count < num_nodes)
            {
                throw std::runtime_error("Typed container count smaller than expected node count." + std::to_string(declared_count) + " < " + std::to_string(num_nodes) + " for marker: " + std::to_string(type_marker));
            }
            if (type_marker == 'D')
            {
                // XGBoost always uses float64 ('D') for double arrays.
                // Bulk-copy the contiguous big-endian block, then byte-swap
                // each 8-byte element in place to host (little-endian) order.
                static_assert(sizeof(double) == 8, "double must be 8 bytes");
                std::memcpy(array, data + pos, num_nodes * 8);
                pos += num_nodes * 8;
                for (uint64_t i = 0; i < num_nodes; i++)
                {
                    uint64_t raw;
                    std::memcpy(&raw, &array[i], 8);
                    raw = bswap64(raw);
                    std::memcpy(&array[i], &raw, 8);
                    array[i] += const_value;
                }
                return;
            }
            // Fallback for other element types (e.g. 'd' float32, 'i' int8).
            for (uint64_t i = 0; i < num_nodes; i++)
            {
                array[i] = readTypedValueAsDouble(type_marker) + const_value;
            }
            return;
        }

        uint64_t start_idx = 0;
        if (first_marker != '#')
        {
            if (first_marker == 'd')
            {
                array[0] = static_cast<double>(readFloat()) + const_value;
            }
            else if (first_marker == 'D')
            {
                array[0] = readDouble() + const_value;
            }
            else if (first_marker == 'i')
            {
                array[0] = static_cast<double>(readInt8()) + const_value;
            }
            else if (first_marker == 'U')
            {
                array[0] = static_cast<double>(readByte()) + const_value;
            }
            else if (first_marker == 'I')
            {
                array[0] = static_cast<double>(readInt16()) + const_value;
            }
            else if (first_marker == 'l')
            {
                array[0] = static_cast<double>(readInt32()) + const_value;
            }
            else if (first_marker == 'L')
            {
                array[0] = static_cast<double>(readInt64()) + const_value;
            }
            else
            {
                throw std::runtime_error("Unsupported marker for value: " + std::to_string(first_marker));
            }
            start_idx = 1;
        }
        else
        {
            uint8_t count_marker = readByte();
            uint64_t declared_count = readNonNegativeIntByMarker(count_marker);
            if (declared_count < num_nodes)
            {
                throw std::runtime_error("Container count smaller than expected node count." + std::to_string(declared_count) + " < " + std::to_string(num_nodes) + " for marker: " + std::to_string(count_marker));
            }
        }

        for (uint64_t i = start_idx; i < num_nodes; i++)
        {
            uint8_t marker = readNextValueMarker();
            if (marker == 'd')
            {
                array[i] = static_cast<double>(readFloat()) + const_value;
            }
            else if (marker == 'D')
            {
                array[i] = readDouble() + const_value;
            }
            else if (marker == 'i')
            {
                array[i] = static_cast<double>(readInt8()) + const_value;
            }
            else if (marker == 'U')
            {
                array[i] = static_cast<double>(readByte()) + const_value;
            }
            else if (marker == 'I')
            {
                array[i] = static_cast<double>(readInt16()) + const_value;
            }
            else if (marker == 'l')
            {
                array[i] = static_cast<double>(readInt32()) + const_value;
            }
            else if (marker == 'L')
            {
                array[i] = static_cast<double>(readInt64()) + const_value;
            }
            else
            {
                throw std::runtime_error("Unsupported marker for value: " + std::to_string(marker));
            }
        }
    }

    void fillArray(int64_t *array, uint64_t num_nodes)
    {
        uint8_t first_marker = readNextValueMarker();

        if (first_marker == '$')
        {
            uint8_t type_marker = readByte();
            uint8_t count_prefix = readByte();
            if (count_prefix != '#')
            {
                throw std::runtime_error("Expected '#' after UBJSON typed container marker '$'.");
            }
            uint8_t count_marker = readByte();
            uint64_t declared_count = readNonNegativeIntByMarker(count_marker);
            if (declared_count < num_nodes)
            {
                throw std::runtime_error("Typed container count smaller than expected node count." + std::to_string(declared_count) + " < " + std::to_string(num_nodes) + " for marker: " + std::to_string(type_marker));
            }
            if (type_marker == 'l')
            {
                // XGBoost always uses int32 ('l') for integer arrays.
                // The destination is int64_t so we can't memcpy the whole block
                // directly, but we can avoid per-element function-call overhead:
                // load 4 bytes with memcpy, byte-swap, sign-extend, advance pos
                // once for the whole array at the end.
                static_assert(sizeof(int32_t) == 4, "int32_t must be 4 bytes");
                for (uint64_t i = 0; i < num_nodes; i++)
                {
                    uint32_t raw;
                    std::memcpy(&raw, data + pos + i * 4, 4);
                    array[i] = static_cast<int64_t>(static_cast<int32_t>(bswap32(raw)));
                }
                pos += num_nodes * 4;
                return;
            }
            // Fallback for other element types (e.g. 'i' int8, 'L' int64).
            for (uint64_t i = 0; i < num_nodes; i++)
            {
                array[i] = readTypedValueAsInt64(type_marker);
            }
            return;
        }

        uint64_t start_idx = 0;
        if (first_marker != '#')
        {
            if (first_marker == 'i')
            {
                array[0] = static_cast<int64_t>(readInt8());
            }
            else if (first_marker == 'U')
            {
                array[0] = static_cast<int64_t>(readByte());
            }
            else if (first_marker == 'I')
            {
                array[0] = static_cast<int64_t>(readInt16());
            }
            else if (first_marker == 'l')
            {
                array[0] = static_cast<int64_t>(readInt32());
            }
            else if (first_marker == 'L')
            {
                array[0] = static_cast<int64_t>(readInt64());
            }
            else
            {
                throw std::runtime_error("Unsupported marker for integer value: " + std::to_string(first_marker));
            }
            start_idx = 1;
        }
        else
        {
            uint8_t count_marker = readByte();
            uint64_t declared_count = readNonNegativeIntByMarker(count_marker);
            if (declared_count < num_nodes)
            {
                throw std::runtime_error("Container count smaller than expected node count." + std::to_string(declared_count) + " < " + std::to_string(num_nodes) + " for marker: " + std::to_string(count_marker));
            }
        }

        for (uint64_t i = start_idx; i < num_nodes; i++)
        {
            uint8_t marker = readNextValueMarker();
            if (marker == 'i')
            {
                array[i] = static_cast<int64_t>(readInt8());
            }
            else if (marker == 'U')
            {
                array[i] = static_cast<int64_t>(readByte());
            }
            else if (marker == 'I')
            {
                array[i] = static_cast<int64_t>(readInt16());
            }
            else if (marker == 'l')
            {
                array[i] = static_cast<int64_t>(readInt32());
            }
            else if (marker == 'L')
            {
                array[i] = static_cast<int64_t>(readInt64());
            }
            else
            {
                throw std::runtime_error("Unsupported marker for integer value: " + std::to_string(marker));
            }
        }
    }

    // skipTo advances past a UBJSON key by scanning for (length_marker + length +
    // key_bytes) and leaves pos right after the key string (ready to read the value).
    // Pass require_object_start=true for base_weights, which is always the first key
    // in a tree object: the byte preceding the length marker must then be '{', which
    // eliminates false positives from binary float/int array payload data.
    // target_key must be a null-terminated string literal whose length == target_length.
    void skipTo(uint64_t target_length, const char *target_key,
                bool require_object_start = false)
    {
        while (pos < size)
        {
            size_t marker_pos = pos;
            uint8_t marker = readByte();
            if (!isIntMarker(marker))
                continue;
            uint64_t length = 0;
            if (!tryReadNonNegativeIntByMarker(marker, length))
                continue;
            if (length != target_length || pos + length > size)
                continue;
            if (std::memcmp(data + pos, target_key, target_length) != 0)
            {
                // Not a match — advance past these bytes and keep scanning.
                pos += target_length;
                continue;
            }
            // Matched. Check structural guard before accepting.
            if (require_object_start &&
                (marker_pos == 0 || data[marker_pos - 1] != static_cast<uint8_t>('{')))
            {
                pos += target_length; // false positive — keep scanning
                continue;
            }
            pos += target_length;
            return;
        }
        throw std::runtime_error(std::string("Key not found: ") + target_key);
    }

    // Skip a single UBJSON scalar integer value (marker byte + data bytes).
    void skipIntegerValue()
    {
        if (pos >= size)
            return;
        uint8_t marker = readByte();
        uint64_t dummy = 0;
        if (!tryReadNonNegativeIntByMarker(marker, dummy))
        {
            // Unexpected marker; just leave pos advanced past the marker byte.
        }
    }

    // Returns the byte size of one element in a UBJSON typed array.
    static size_t typedElementSize(uint8_t type_marker)
    {
        switch (type_marker)
        {
        case 'T':
        case 'F':
        case 'Z':
        case 'i':
        case 'U':
            return 1;
        case 'I':
            return 2;
        case 'l':
        case 'd':
            return 4;
        case 'L':
        case 'D':
            return 8;
        default:
            throw std::runtime_error("typedElementSize: unknown UBJSON type: " + std::to_string(type_marker));
        }
    }

    // Consumes a single UBJSON scalar or nested container value starting at pos.
    // Used by skipArrayValue when handling untyped count-prefixed arrays.
    void skipSingleValue()
    {
        if (pos >= size)
            return;
        uint8_t marker = readByte();
        switch (marker)
        {
        case '[':
            pos--;
            skipArrayValue();
            return;
        case '{':
        {
            size_t depth = 1;
            while (pos < size && depth > 0)
            {
                uint8_t b = readByte();
                if (b == '{')
                    depth++;
                else if (b == '}')
                    depth--;
            }
            return;
        }
        case 'T':
        case 'F':
        case 'Z':
            return; // no payload
        case 'i':
        case 'U':
            pos += 1;
            return; // 1-byte payload
        case 'I':
            pos += 2;
            return; // 2-byte payload
        case 'l':
        case 'd':
            pos += 4;
            return; // 4-byte payload
        case 'L':
        case 'D':
            pos += 8;
            return; // 8-byte payload
        case 'S':
        case 'H':
        {
            uint8_t lm = readByte();
            uint64_t len = readNonNegativeIntByMarker(lm);
            pos += len;
            return;
        }
        default:
            throw std::runtime_error("skipSingleValue: unknown marker: " + std::to_string(marker));
        }
    }

    // Consumes a complete UBJSON array value, including the opening '[',
    // starting at the current byte.  Handles all three array forms:
    //   typed       [$T#count data...]   — XGBoost always writes this form
    //   count-only  [#count elem...]     — fallback
    //   plain       [elem... ]           — fallback
    void skipArrayValue()
    {
        if (pos >= size || data[pos] != '[')
            return;
        pos++; // consume '['
        if (pos >= size)
            return;
        if (data[pos] == '$')
        {
            pos++; // consume '$'
            uint8_t type_marker = readByte();
            if (pos >= size || data[pos] != '#')
                throw std::runtime_error("skipArrayValue: expected '#' after '$'");
            pos++; // consume '#'
            uint8_t count_marker = readByte();
            uint64_t count = readNonNegativeIntByMarker(count_marker);
            pos += count * typedElementSize(type_marker);
        }
        else if (data[pos] == '#')
        {
            pos++; // consume '#'
            uint8_t count_marker = readByte();
            uint64_t count = readNonNegativeIntByMarker(count_marker);
            for (uint64_t j = 0; j < count; j++)
                skipSingleValue();
        }
        else
        {
            // Plain array: scan for matching ']', tracking nesting depth.
            size_t depth = 1;
            while (pos < size && depth > 0)
            {
                uint8_t b = readByte();
                if (b == '[')
                    depth++;
                else if (b == ']')
                    depth--;
            }
        }
    }

    ParsedForest extractTreeStructure(int class_label)
    {
        // This methods parses the data, until it finds the "trees" key.
        // After that, it parses the tree structure into plain C++ buffers.
        int64_t num_class = readNumClassOrOne();

        // XGBoost stores num_class * num_rounds trees in round-robin order:
        // tree i belongs to class (i % num_class).
        uint64_t total_trees = static_cast<uint64_t>(readNumTrees());
        uint64_t num_rounds = total_trees / static_cast<uint64_t>(num_class > 0 ? num_class : 1);
        double base_score = readBaseScoreOrZero(class_label);

        // Whether to filter: keep only trees for the requested class.
        bool filtering = (class_label >= 0) && (num_class > 1);

        // Each included tree carries base_score / num_rounds so that the sum
        // across all rounds equals base_score for the selected class.
        // For regression/binary (num_class==1), num_rounds == total_trees.
        double base_score_per_tree = base_score / static_cast<double>(num_rounds == 0 ? 1 : num_rounds);

        skipToTrees();
        skipArrayHeader(); // consume '[' and optional '#count'/'$type#count' prefix

        ParsedForest forest;
        forest.num_class = num_class;
        forest.base_score = base_score;
        forest.trees.reserve(filtering ? num_rounds : total_trees);

        for (uint64_t i = 0; i < total_trees; i++)
        {
            // Advance main stream to this tree's base_weights key.
            // For excluded trees we continue immediately; the next iteration's
            // skipTo will find the following tree's base_weights from inside
            // this tree's data — no need to parse the remaining fields.
            skipTo(12, "base_weights", true);

            bool include = !filtering || ((int)(i % static_cast<uint64_t>(num_class)) == class_label);
            if (!include)
                continue;

            // pos is now at this tree's values array '['.
            // Use a copy for filling so the main stream position is only
            // advanced once after all fields have been consumed.
            ByteStream treeStream = *this;

            // Peek the node count from the array header via a cheap struct copy,
            // then fill from the same position — no redundant scan.
            ByteStream countPeek = treeStream;
            uint64_t num_nodes = countPeek.readArrayCount();

            ParsedTreeArrays tree;
            tree.node_ids.resize(num_nodes);
            tree.feature_ids.resize(num_nodes);
            tree.thresholds.resize(num_nodes);
            tree.values.resize(num_nodes);
            tree.left_children.resize(num_nodes);
            tree.right_children.resize(num_nodes);
            tree.default_children.resize(num_nodes);
            tree.node_sample_weights.resize(num_nodes);

            // node_ids are simply 0..num_nodes-1
            for (uint64_t j = 0; j < num_nodes; j++)
                tree.node_ids[j] = static_cast<int64_t>(j);

            // Process all XGBoost tree keys in their known alphabetical order.
            // Each skipTo call starts immediately after the previous value, so
            // the scan covers only a handful of structural bytes — false
            // positives from binary array payload data are impossible.

            // base_weights → leaf/split values (stream already positioned at '[')
            treeStream.fillArray(tree.values.data(), num_nodes, base_score_per_tree);

            // categories group (empty for non-categorical models — skip all)
            treeStream.skipTo(10, "categories");
            treeStream.skipArrayValue();
            treeStream.skipTo(16, "categories_nodes");
            treeStream.skipArrayValue();
            treeStream.skipTo(19, "categories_segments");
            treeStream.skipArrayValue();
            treeStream.skipTo(16, "categories_sizes");
            treeStream.skipArrayValue();

            // default_left → default-branch indicator per node
            treeStream.skipTo(12, "default_left");
            treeStream.fillArray(tree.default_children.data(), num_nodes);

            // id: integer == tree index; skip it so its value bytes cannot
            // be misread as the length of the following 'left_children' key.
            treeStream.skipTo(2, "id");
            treeStream.skipIntegerValue();

            treeStream.skipTo(13, "left_children");
            treeStream.fillArray(tree.left_children.data(), num_nodes);

            // loss_changes, parents: not used by the converter
            treeStream.skipTo(12, "loss_changes");
            treeStream.skipArrayValue();
            treeStream.skipTo(7, "parents");
            treeStream.skipArrayValue();

            treeStream.skipTo(14, "right_children");
            treeStream.fillArray(tree.right_children.data(), num_nodes);

            // split_conditions → thresholds
            treeStream.skipTo(16, "split_conditions");
            treeStream.fillArray(tree.thresholds.data(), num_nodes);

            // split_indices → feature IDs
            treeStream.skipTo(13, "split_indices");
            treeStream.fillArray(tree.feature_ids.data(), num_nodes);

            // split_type: not used
            treeStream.skipTo(10, "split_type");
            treeStream.skipArrayValue();

            // sum_hessian → node sample weights
            treeStream.skipTo(11, "sum_hessian");
            treeStream.fillArray(tree.node_sample_weights.data(), num_nodes);

            // tree_param and closing '}' are left unconsumed; advancing pos to
            // treeStream.pos places the main stream right after sum_hessian,
            // which is the correct starting point for the next tree's '{'.
            pos = treeStream.pos;

            forest.trees.push_back(std::move(tree));
        }
        return forest;
    };
};

class StringStream
{
private:
    const char *data;
    size_t pos;
    size_t size;

public:
    StringStream(const char *str, size_t len) : data(str), pos(0), size(len) {}

    void skipWhitespace()
    {
        // Skip any whitespace characters starting from the current position. https://en.cppreference.com/w/cpp/string/byte/isspace.html
        // Also skips \t and \n, which can appear in the String
        while (pos < size && std::isspace(static_cast<unsigned char>(data[pos])))
        {
            pos++;
        }
    }
    void skipLine()
    {
        // Skip characters until the end of the current line (i.e., until a newline character or the end of the string).
        while (pos < size && data[pos] != '\n')
        {
            pos++;
        }
        if (pos < size && data[pos] == '\n')
        {
            pos++; // Skip the newline character as well
        }
    }

    int remapNodeId(int n_internals, int id)
    {
        if (id >= 0)
            return id;               // Leave internal node IDs unchanged
        return (n_internals)-id - 1; // Remap leaf node IDs to be after internal nodes
    }

    void findKey(const char *key)
    {
        // Find the given key in the string, starting from the current position.
        // Advances pos to the character immediately following the key if found.
        size_t key_len = std::strlen(key);
        while (pos < size)
        {
            skipWhitespace();
            if (pos + key_len <= size && std::memcmp(data + pos, key, key_len) == 0)
            {
                pos += key_len;
                return;
            }
            pos++;
        }
        throw std::runtime_error(std::string("Key not found: ") + key);
    }

    bool tryFindKey(const char *key)
    {
        // Find the given key in the string, starting from the current position.
        // Advances pos to the character immediately following the key if found.
        size_t key_len = std::strlen(key);
        while (pos < size)
        {
            skipWhitespace();
            if (pos + key_len <= size && std::memcmp(data + pos, key, key_len) == 0)
            {
                pos += key_len;
                return true;
            }
            pos++;
        }
        return false;
    }

    void parseIntLine(std::vector<int64_t> &output)
    {
        // Parse a line of whitespace-separated integers starting from the current position and append them to the output vector.
        while (pos < size && data[pos] != '\n')
        {
            skipWhitespace();
            if (pos >= size || data[pos] == '\n')
                break;
            char *endptr = nullptr;
            int64_t value = std::strtoll(data + pos, &endptr, 10);
            if (endptr == data + pos)
            {
                throw std::runtime_error("Expected integer at position: " + std::to_string(pos));
            }
            output.push_back(value);
            pos = endptr - data; // Advance pos to the character after the parsed integer. The difference computes the new position relative to the start of the string.
        }
    }

    void parseDoubleLine(std::vector<double> &output)
    {
        // Parse a line of whitespace-separated floating-point numbers starting from the current position and append them to the output vector.
        while (pos < size && data[pos] != '\n')
        {
            skipWhitespace();
            if (pos >= size || data[pos] == '\n')
                break;
            char *endptr = nullptr;
            double value = std::strtod(data + pos, &endptr);
            if (endptr == data + pos)
            {
                throw std::runtime_error("Expected floating-point number at position: " + std::to_string(pos));
            }
            output.push_back(value);
            pos = endptr - data; // Advance pos to the character after the parsed number. The difference computes the new position relative to the start of the string.
        }
    }

    void parseIntSingle(int &value)
    {
        skipWhitespace();
        char *endptr = nullptr;
        value = std::strtol(data + pos, &endptr, 10);
        pos = endptr - data; // Advance pos to the character after the parsed integer. The difference computes the new position relative to the start of the string.
    }

    // Read num_tree_per_iteration from the header (equals num_class for multiclass).
    // Uses a copy so the main stream position is unchanged.
    int readNumTreePerIteration()
    {
        StringStream s = *this;
        size_t key_len = std::strlen("num_tree_per_iteration=");
        while (s.pos < s.size)
        {
            s.skipWhitespace();
            if (s.pos + key_len <= s.size && std::memcmp(s.data + s.pos, "num_tree_per_iteration=", key_len) == 0)
            {
                s.pos += key_len;
                int val = 1;
                s.parseIntSingle(val);
                return val;
            }
            s.pos++;
        }
        return 1;
    }

    ParsedForest extractTreeStructure(int class_label = -1)
    {
        int num_class = readNumTreePerIteration();
        bool filtering = (class_label >= 0) && (num_class > 1);

        ParsedForest forest;
        forest.num_class = static_cast<int64_t>(num_class);
        int tree_id = 0, num_cat = 0, num_leaves = 0, num_nodes = 0, num_internal = 0;

        while (true)
        {
            // Find next tree; pos advances past "Tree=".
            // For excluded trees we continue immediately — tryFindKey will
            // scan forward from inside this tree's text to the next "Tree=".
            if (!tryFindKey("Tree="))
                break;
            parseIntSingle(tree_id);

            bool include = !filtering || (tree_id % num_class == class_label);
            if (!include)
                continue;

            std::vector<int64_t> node_ids, feature_ids, left_children, right_children, default_children, internal_count, decision_types;
            std::vector<double> thresholds, values, leaf_count, internal_values;
            // Get num_leaves
            findKey("num_leaves=");
            parseIntSingle(num_leaves);
            num_internal = num_leaves - 1;
            num_nodes = num_internal + num_leaves; //
            // Get num categorical features (0 for non-categorical models). NOTICE: Currently not support for categorical features.
            findKey("num_cat=");
            parseIntSingle(num_cat);
            // Extract Split feature IDs
            findKey("split_feature=");
            parseIntLine(feature_ids);
            // Skip split_gain line
            skipLine();
            // Extract thresholds
            findKey("threshold=");
            parseDoubleLine(thresholds);
            // Extract decision_type (bitset-encoded split metadata)
            findKey("decision_type=");
            parseIntLine(decision_types);
            // Extract left_children
            findKey("left_child=");
            parseIntLine(left_children);
            // Extract right_children
            findKey("right_child=");
            parseIntLine(right_children);
            // Extract Leaf_values
            findKey("leaf_value=");
            parseDoubleLine(values);
            // Skip leaf weight
            skipLine();
            // Extract node_sample_weights
            findKey("leaf_count=");
            parseDoubleLine(leaf_count);
            // Extract internal node values
            findKey("internal_value=");
            parseDoubleLine(internal_values);
            // Skip Internal weight
            skipLine();
            // Get Internal count
            findKey("internal_count=");
            parseIntLine(internal_count);
            // Skip is_linear
            skipLine();
            // Skip shrinkage value
            skipLine();

            ParsedTreeArrays tree;
            tree.node_ids.resize(num_nodes, 0);
            tree.feature_ids.resize(num_nodes, -1);
            tree.thresholds.resize(num_nodes, 0.0);
            tree.values.resize(num_nodes, 0.0);
            tree.left_children.resize(num_nodes, -1);
            tree.right_children.resize(num_nodes, -1);
            tree.default_children.resize(num_nodes, -1);
            tree.node_sample_weights.resize(num_nodes, 0.0);

            if (feature_ids.size() < static_cast<size_t>(num_internal) ||
                thresholds.size() < static_cast<size_t>(num_internal) ||
                left_children.size() < static_cast<size_t>(num_internal) ||
                right_children.size() < static_cast<size_t>(num_internal) ||
                internal_count.size() < static_cast<size_t>(num_internal) ||
                decision_types.size() < static_cast<size_t>(num_internal) ||
                internal_values.size() < static_cast<size_t>(num_internal) ||
                values.size() < static_cast<size_t>(num_leaves) ||
                leaf_count.size() < static_cast<size_t>(num_leaves))
            {
                throw std::runtime_error("LightGBM tree field length mismatch");
            }

            // Process the internal nodes. We can just copy the split feature IDs, thresholds, left/right children, and sample weights. The default child is always the left child in LightGBM.
            for (int i = 0; i < num_internal; i++)
            {
                tree.node_ids[i] = i;
                tree.feature_ids[i] = feature_ids[i];
                tree.thresholds[i] = thresholds[i];
                tree.values[i] = internal_values[i];
                tree.left_children[i] = remapNodeId(num_internal, left_children[i]);
                tree.right_children[i] = remapNodeId(num_internal, right_children[i]);
                bool default_left = (decision_types[i] & 2) != 0;
                tree.default_children[i] = default_left
                                               ? remapNodeId(num_internal, left_children[i])
                                               : remapNodeId(num_internal, right_children[i]);
                tree.node_sample_weights[i] = internal_count[i];
            }
            // Process the leaf nodes.
            for (int i = num_internal; i < num_nodes; i++)
            {
                tree.node_ids[i] = i;
                tree.values[i] = values[i - num_internal];
                tree.node_sample_weights[i] = leaf_count[i - num_internal];
            }

            if (include)
                forest.trees.push_back(std::move(tree));
        }
        return forest;
    }
};
