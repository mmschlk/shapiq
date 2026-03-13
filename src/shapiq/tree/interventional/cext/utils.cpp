#include <cstdint>
#include <vector>
#ifndef UTILS_H
#define UTILS_H

#ifdef _MSC_VER
#include <intrin.h>
static inline int ctz64(uint64_t x) {
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
}
#else
static inline int ctz64(uint64_t x) {
    return __builtin_ctzll(x);
}
#endif

enum class IndexType
{
    SII,
    BII,
    CHII,
    FBII,
    FSII,
    STII,
    CUSTOM
};

enum class StorageType
{
    NUM,
    STACK,
    HEAP
};

enum DecisionType
{
    LESS_EQUAL = 0,
    LESS_THAN = 1
};


class BitSet
{
    /**
     * This class implements a simple bitset to efficiently store sets of features.
     *  It uses a single uint64_t for small feature sets (up to 64 features) and a vector of uint64_t for larger feature sets.
     *  The contains and add methods allow us to check if a feature is in the set and to add a feature to the set, respectively.
     */
    uint64_t size;
public:
    // explicit constructor to initialize the BitSet with the number of features. It determines whether to use small storage (a single uint64_t) or large storage (a vector of uint64_t) based on the number of features.

    BitSet() : size(0), num_features(0), small_data(0ULL),  storage_type(StorageType::NUM) {}

    explicit BitSet(int64_t num_features)
        : num_features(num_features), size(0)
    {
        if (num_features <= 64) {
            storage_type = StorageType::NUM;
            small_data = 0ULL;
        } else if  (num_features <= 256) {
            storage_type = StorageType::STACK;
            std::fill(std::begin(small_buffer), std::end(small_buffer), 0ULL);
        } else {
            storage_type = StorageType::HEAP;
            data.assign((num_features + 63) / 64, 0ULL);
        }

    }

    bool contains(int64_t feature_id) const
    {
        // (feature_id & 63) is equivalent to feature_id % 64, which gives us the position of the bit within the uint64_t word.
        const uint64_t mask = 1ULL << (feature_id & 63);
        if (storage_type == StorageType::NUM)
        {
            // For small feature sets, we can directly check the bit in the small_data uint64_t.
            return (small_data & mask) != 0ULL;
        }
        // feature_id >> 6 is equivalent to feature_id / 64, which gives us the index of the uint64_t word in the data vector that contains the bit for the given feature_id.
        const size_t word_index = feature_id >> 6;

        const uint64_t word = (storage_type == StorageType::STACK) ? small_buffer[word_index] : data[word_index];

        return (word & mask) != 0ULL;

    }

    void from_array(const int64_t *feature_ids, size_t count)
    {
        for (size_t i = 0; i < count; i++)
        {
            if (feature_ids[i] < 0)
            {
                // Assume that -1 is used as a sentinel value to indicate the end of the feature list. If we encounter a negative feature ID, we stop processing further.
                break;
            }
            add(feature_ids[i]);
        }
    }

    bool add(int64_t feature_id)
    {
        // (feature_id & 63) is equivalent to feature_id % 64, which gives us the position of the bit within the uint64_t word.
        const uint64_t mask = 1ULL << (feature_id & 63);
        if (storage_type == StorageType::NUM)
        {
            const bool was_present = (small_data & mask) != 0ULL;
            small_data |= mask;
            size += was_present ? 0 : 1;
            return !was_present;
        }
        const size_t word_index = feature_id >> 6;
        uint64_t &word = (storage_type == StorageType::STACK) ? small_buffer[word_index] : data[word_index];
        const bool was_present = (word & mask) != 0ULL;
        word |= mask;
        size += was_present ? 0 : 1;
        return !was_present;
    }

    bool remove(int64_t feature_id)
    {
        // (feature_id & 63) is equivalent to feature_id % 64, which gives us the position of the bit within the uint64_t word.
        const uint64_t mask = 1ULL << (feature_id & 63);
        if (storage_type == StorageType::NUM)
        {
            // Remove the feature by clearing the corresponding bit in small_data using bitwise AND with the negation of the mask.
            const bool was_present = (small_data & mask) != 0ULL;
            small_data &= ~mask;
            size -= was_present ? 1 : 0;
            return was_present;
        }
        // feature_id >> 6 is equivalent to feature_id / 64, which gives us the index of the uint64_t word in the data vector that contains the bit for the given feature_id.
        const size_t word_index = feature_id >> 6;
        uint64_t &word = (storage_type == StorageType::STACK) ? small_buffer[word_index] : data[word_index];
        const bool was_present = (word & mask) != 0ULL;
        word &= ~mask;
        size -= was_present ? 1 : 0;
        return was_present;
    }

    // Create a function using a template to apply a given function to each set bit in the BitSet.
    // This allows us to efficiently iterate over the features in the set without to copy the entire BitSet.
    // Here `template<typename Func>` allows us to define a function that can accept any callable type (e.g., function pointer, lambda, functor) as an argument.
    // The compiler will generate the appropriate code for the specific type of function that is passed in when the for_each_set_bit function is called.
    // This provides flexibility and allows us to use different types of functions without having to write separate code for each type.
    template<typename Func>
    void for_each_set_bit(Func&& f) const {
        if (storage_type == StorageType::NUM)
        {
            uint64_t bits = small_data;
            while (bits)
            {
                int feature_id = ctz64(bits); // Get the index of the least significant set bit
                f(feature_id);
                bits &= bits - 1; // Clear the least significant set bit
            }
        }
        else if (storage_type == StorageType::STACK)
        {
            for (size_t i = 0; i < 4; i++)
            {
                uint64_t bits = small_buffer[i];
                while (bits)
                {
                    int feature_id = ctz64(bits) + (i * 64); // Get the index of the least significant set bit and adjust for the word index
                    f(feature_id);
                    bits &= bits - 1; // Clear the least significant set bit
                }
            }
        } else
        {
            for (size_t i = 0; i < data.size(); i++)
            {
                uint64_t bits = data[i];
                while (bits)
                {
                    int feature_id = ctz64(bits) + (i * 64); // Get the index of the least significant set bit and adjust for the word index
                    f(feature_id);
                    bits &= bits - 1; // Clear the least significant set bit
                }
            }
        }
    }
    uint64_t num_bits() const {
        return size;
    }

    bool equals(const BitSet &other) const
    {
        // Two BitSets are equal if they have the same number of features, the same size (number of set bits), and contain the same features.
        // For small feature sets, we can directly compare the small_data uint64_t.
        // For larger feature sets, we need to compare the contents of the data vector.
        if (this->num_features != other.num_features || this->size != other.size)
        {
            return false;
        }
        if (this->storage_type == StorageType::NUM && other.storage_type == StorageType::NUM)
        {
            return this->small_data == other.small_data;
        }
        for (int64_t feature_id = 0; feature_id < num_features; ++feature_id)
        {
            if (this->contains(feature_id) != other.contains(feature_id))
            {
                return false;
            }
        }
        return true;
    }

    std::size_t hash() const
    {
        // We compute a hash value for the BitSet by combining the number of features, the size (number of set bits), and the indices of the set bits. This allows us to use BitSet as a key in hash-based data structures like unordered_map.
        std::size_t seed = 1469598103934665603ULL;
        auto hash_combine = [&](std::size_t value)
        {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        hash_combine(std::hash<int64_t>{}(num_features));
        hash_combine(std::hash<uint64_t>{}(size));
        this->for_each_set_bit([&](uint64_t feature_id)
                               { hash_combine(std::hash<uint64_t>{}(feature_id)); });
        return seed;
    }

    void fill_buffer(uint64_t* buffer) const {
        size_t idx = 0;
        this->for_each_set_bit([&](uint64_t feature_id) {
            buffer[idx++] = feature_id;
        });
    }

private:
    int64_t num_features;
    uint64_t small_data;
    uint64_t small_buffer[4]; // Buffer for up to 256 features (4 * 64 bits)
    std::vector<uint64_t> data;
    StorageType storage_type;
};


class Tree
{
public:
    Tree(float *leaf_predictions,
         float *thresholds,
         int64_t *features,
         int64_t *children_left,
         int64_t *children_right,
         bool *children_missing,
         std::string decision_type)
    {
        this->leaf_predictions = leaf_predictions;
        this->thresholds = thresholds;
        this->features = features;
        this->children_left = children_left;
        this->children_right = children_right;
        this->children_missing = children_missing;

        this->decision_type = decision_type == "<=" ? DecisionType::LESS_EQUAL : DecisionType::LESS_THAN;
    }
    DecisionType decision_type;
    float *leaf_predictions;
    float *thresholds;
    int64_t *features;
    int64_t *children_left;
    int64_t *children_right;
    bool *children_missing;

    bool is_leaf(int64_t node_id) const
    {
        return this->children_left[node_id] == -1 && this->children_right[node_id] == -1;
    }

    bool goes_left(float feature_value, int64_t node_id)
    {
        if (std::isnan(feature_value))
        {
            return this->children_missing[node_id];
        }

        if (this->decision_type == DecisionType::LESS_EQUAL)
        {
            return feature_value <= this->thresholds[node_id];
        }
        else if (this->decision_type == DecisionType::LESS_THAN)
        {
            return feature_value < this->thresholds[node_id];
        }
        else
        {
            throw std::invalid_argument("Unsupported decision type: " + std::to_string(static_cast<int>(this->decision_type)));
        }
        return feature_value <= this->thresholds[node_id];
    }
};


class StackFrame
{
public:
    StackFrame(int64_t node_id, const BitSet &E, const BitSet &R, uint64_t e = 0, uint64_t r = 0)
        : node_id(node_id), E(E), R(R), e(e), r(r)
    {
    }
    int64_t node_id;
    BitSet E;   // Features used by point to explain
    BitSet R;   // Features used by reference point
    uint64_t e; // Number of features in E
    uint64_t r; // Number of features in R
};

struct BitSetHash
{
    std::size_t operator()(const BitSet &set) const
    {
        return set.hash();
    }
};

struct BitSetEqual
{
    bool operator()(const BitSet &lhs, const BitSet &rhs) const
    {
        return lhs.equals(rhs);
    }
};




#endif
