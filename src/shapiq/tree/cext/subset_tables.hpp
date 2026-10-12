// The tables of interactions and their hash index, shared by the interventional and the
// quadrature extension.
//
// An interaction can only be non-zero if all its feature ids occur on one root-to-leaf path.
// Per order s >= 2 the kernels keep a sorted table of exactly those interactions, each entry
// listing its feature ids in ascending order, and map an interaction met at a leaf to its
// position in that table. The output array is the order-1 block [0, n_features) followed by
// one block per order with one entry per table entry, at the same position.
//
// Contents: Key128, an interaction's 128-bit key (its feature ids as digits in base
// n_features); Geometry, every offset derived from the per-order counts; SubsetIndex, the hash
// index from key to position; TableSearch, the search of a table when no index exists;
// SubsetCollector, one DFS per tree building the tables and the index; and the Python bindings
// preprocess_subset_tables and layout_to_dict.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

namespace subset_tables
{
    // Hash multiplier, 2^64 / golden ratio.
    constexpr uint64_t kIndexHashMultiplier = 0x9E3779B97F4A7C15ULL;

    // An interaction's 128-bit key: its ascending feature ids as digits in base n_features.
    // An index is only built while every key stays below 2^127 (keys_fit), so hi never equals
    // kEmptyKey.
    struct Key128
    {
        uint64_t hi = 0;
        uint64_t lo = 0;
    };
    constexpr uint64_t kEmptyKey = ~uint64_t(0);
    inline bool operator==(Key128 a, Key128 b) { return a.hi == b.hi && a.lo == b.lo; }
    inline bool operator!=(Key128 a, Key128 b) { return !(a == b); }
    inline bool operator<(Key128 a, Key128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }

    // key * n + f for n, f < 2^32; the result must stay below 2^127.
    inline Key128 key_mul_add(Key128 k, uint64_t n, uint64_t f)
    {
#if defined(__SIZEOF_INT128__)
        const unsigned __int128 v = ((static_cast<unsigned __int128>(k.hi) << 64) | k.lo) * n + f;
        return Key128{static_cast<uint64_t>(v >> 64), static_cast<uint64_t>(v)};
#else
        const uint64_t lo_lo = k.lo & 0xffffffffULL, lo_hi = k.lo >> 32;
        const uint64_t p0 = lo_lo * n + f;
        const uint64_t p1 = lo_hi * n + (p0 >> 32);
        return Key128{k.hi * n + (p1 >> 32), (p1 << 32) | (p0 & 0xffffffffULL)};
#endif
    }

    // key /= n (n < 2^32); returns key % n.
    inline uint64_t key_divmod(Key128 &k, uint64_t n)
    {
#if defined(__SIZEOF_INT128__)
        unsigned __int128 v = (static_cast<unsigned __int128>(k.hi) << 64) | k.lo;
        const uint64_t rem = static_cast<uint64_t>(v % n);
        v /= n;
        k = Key128{static_cast<uint64_t>(v >> 64), static_cast<uint64_t>(v)};
        return rem;
#else
        uint64_t r = k.hi % n;
        const uint64_t q_hi = k.hi / n;
        const uint64_t top = (r << 32) | (k.lo >> 32);
        const uint64_t q1 = top / n;
        r = top % n;
        const uint64_t bot = (r << 32) | (k.lo & 0xffffffffULL);
        const uint64_t q0 = bot / n;
        r = bot % n;
        k = Key128{q_hi, (q1 << 32) | q0};
        return r;
#endif
    }

    // Hash of a key; a probe of the index starts at its top bits.
    inline uint64_t key_hash(Key128 k)
    {
        return (k.lo ^ (k.hi * kIndexHashMultiplier)) * kIndexHashMultiplier;
    }

    // Whether n_features^max_order stays below 2^127, i.e. whether every key fits.
    inline bool keys_fit(int64_t n_features, int max_order)
    {
        const uint64_t n = static_cast<uint64_t>(n_features);
        Key128 cap{0, 1};
        for (int i = 0; i < max_order; ++i)
        {
            if (cap.hi + 1 > ((uint64_t(1) << 63) - 1) / n)
                return false;
            cap = key_mul_add(cap, n, 0);
        }
        return true;
    }

    // C(n, k) as int64, or -1 when it does not fit.
    inline int64_t binom_int64(int64_t n, int k)
    {
        if (k < 0 || n < k)
            return 0;
        long double value = 1.0L;
        for (int i = 0; i < k; ++i)
            value = value * static_cast<long double>(n - i) / static_cast<long double>(i + 1);
        return value < 9.0e18L ? static_cast<int64_t>(value + 0.5L) : -1;
    }

    // Whether a table of `count` interactions of `order` holds every interaction of that order
    // (orders 2 and 3 only); positions in a complete table are lexicographic ranks.
    inline bool table_is_complete(int64_t count, int order, int64_t n_features)
    {
        return order <= 3 && count > 0 && count == binom_int64(n_features, order);
    }

    // log2 of the number of index slots for a table of `count` interactions: the smallest power
    // of two with at least 2 * count slots, at least 2. A complete table keeps an empty 2-slot block.
    inline int block_bits(int64_t count, bool complete = false)
    {
        if (complete)
            return 1;
        int bits = 1;
        while ((int64_t(1) << bits) < 2 * count)
            ++bits;
        return bits;
    }

    // Every offset of the layout, derived from the per-order interaction counts. Each vector has
    // one entry per order (entries 0 and 1 are unused).
    struct Geometry
    {
        std::vector<int64_t> table_starts;    // per order: offset of its table in `keys` (in feature ids)
        std::vector<int64_t> block_starts;    // per order: first slot of its index block
        std::vector<int64_t> block_shifts;    // per order: 64 - block_bits(count)
        std::vector<int64_t> output_offsets;  // per order: start of its block in the output array
        int64_t n_keys = 0;                   // length of `keys`
        int64_t n_slots = 0;                  // number of index slots
        int64_t n_outputs = 0;                // length of the output array
    };

    // `counts` has max_order + 1 entries. Orders below max(min_order, 2) own no output block,
    // but their tables and index blocks still exist.
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
            const int bits = block_bits(count, table_is_complete(count, order, n_features));
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

    // Key of the interaction formed by inserting `feature` into the ascending `chosen[0, size)`
    // (which does not contain it), without materializing the feature ids.
    template <typename Int>
    inline Key128 merged_key(const Int *chosen, int size, Int feature, int64_t n_features)
    {
        Key128 key;
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
            key = key_mul_add(key, static_cast<uint64_t>(n_features), static_cast<uint64_t>(current));
        }
        return key;
    }

    // The same insertion, materialized: `out` receives the size + 1 ascending feature ids.
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

    // The hash index: per order a block of slots holding (key, position), probed linearly from
    // the top bits of key_hash. `starts` and `shifts` point into a Geometry.
    struct SubsetIndex
    {
        const int64_t *keys = nullptr;    // per slot the key's (hi, lo) words interleaved; hi == -1 marks an empty slot
        const int32_t *rows = nullptr;    // per slot: the interaction's position in its table
        const int64_t *starts = nullptr;  // per order: first slot of its block
        const int64_t *shifts = nullptr;  // per order: 64 - log2(block size)

        // Position of the order-s interaction with this key, or -1 when it is not in the table.
        // The probe is bounded by the block size.
        int64_t row(Key128 key, int s, const int64_t *counts) const
        {
            const int64_t block = starts[s];
            const int shift = static_cast<int>(shifts[s]);
            const uint64_t mask = (uint64_t(1) << (64 - shift)) - 1;
            uint64_t slot = key_hash(key) >> shift;
            for (uint64_t probe = 0; probe <= mask; ++probe)
            {
                const uint64_t hi = static_cast<uint64_t>(keys[2 * (block + slot)]);
                if (hi == kEmptyKey)
                    return -1;
                if (hi == key.hi && static_cast<uint64_t>(keys[2 * (block + slot) + 1]) == key.lo)
                {
                    const int64_t r = rows[block + slot];
                    return (r >= 0 && r < counts[s]) ? r : -1;
                }
                slot = (slot + 1) & mask;
            }
            return -1;
        }
    };

    // Search of a table when no hash index exists. One leaf event (or one quadrature edge)
    // emits its interactions of an order in increasing order, so the search gallops forward
    // from a per-order cursor and bisects the bracket it finds.
    struct TableSearch
    {
        const int32_t *keys = nullptr;    // the tables (feature ids)
        const int64_t *starts = nullptr;  // per order: start of its table in `keys` (in feature ids)
        const int64_t *counts = nullptr;  // per order: entries in its table
        std::vector<int64_t> cursor;      // per order: position after the last entry found

        TableSearch() = default;
        TableSearch(const int32_t *keys_, const int64_t *starts_, const int64_t *counts_, int max_order)
            : keys(keys_), starts(starts_), counts(counts_), cursor(static_cast<size_t>(max_order) + 1, 0)
        {
        }

        // Lexicographic compare of a table entry against an interaction (both of order s).
        template <typename Int>
        static int compare(const int32_t *entry, const Int *features, int s)
        {
            for (int i = 0; i < s; ++i)
            {
                if (entry[i] != static_cast<int32_t>(features[i]))
                    return (entry[i] < static_cast<int32_t>(features[i])) ? -1 : 1;
            }
            return 0;
        }

        // Position of the order-s interaction with these ascending feature ids, or -1 when it
        // is not in the table.
        template <typename Int>
        int64_t position(const Int *features, int s)
        {
            const int32_t *base = keys + starts[s];
            const int64_t count = counts[s];
            int64_t lo = cursor[s];
            if (lo >= count || compare(base + lo * s, features, s) > 0)
                lo = 0;
            int64_t bound = 1;  // exponential gallop to a bracket whose upper entry is >= features
            while (lo + bound < count && compare(base + (lo + bound) * s, features, s) < 0)
                bound <<= 1;
            int64_t hi = std::min(lo + bound + 1, count);
            lo = lo + (bound >> 1);
            while (lo < hi)  // bisect the bracket
            {
                const int64_t mid = lo + ((hi - lo) >> 1);
                if (compare(base + mid * s, features, s) < 0)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo >= count || compare(base + lo * s, features, s) != 0)
                return -1;
            cursor[s] = lo + 1;
            return lo;
        }
    };

    // Open-addressing set of keys used while collecting (linear probing, at most half full).
    struct FlatKeySet
    {
        std::vector<Key128> slots;
        uint64_t mask = 0;
        int shift = 63;
        size_t size = 0;

        FlatKeySet() { reset(1024); }

        void reset(size_t capacity)
        {
            int bits = 1;
            while ((size_t(1) << bits) < capacity)
                ++bits;
            slots.assign(size_t(1) << bits, Key128{kEmptyKey, 0});
            mask = (uint64_t(1) << bits) - 1;
            shift = 64 - bits;
            size = 0;
        }

        void insert(Key128 key)
        {
            uint64_t slot = key_hash(key) >> shift;
            while (true)
            {
                const Key128 stored = slots[slot];
                if (stored == key)
                    return;
                if (stored.hi == kEmptyKey)
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
            std::vector<Key128> old = std::move(slots);
            reset(old.size() * 2);
            for (Key128 key : old)
                if (key.hi != kEmptyKey)
                    insert(key);
        }

        std::vector<Key128> sorted_keys() const
        {
            std::vector<Key128> keys;
            keys.reserve(size);
            for (Key128 key : slots)
                if (key.hi != kEmptyKey)
                    keys.push_back(key);
            std::sort(keys.begin(), keys.end());
            return keys;
        }
    };

    // The tables of an ensemble, plus their hash index when the keys fit.
    struct Tables
    {
        std::vector<int32_t> keys;    // the tables back to back, each entry its `order` feature ids
        std::vector<int64_t> counts;  // per order: number of entries
        bool has_index = false;       // false when a key would overflow 127 bits
        std::vector<int64_t> slot_keys; // per slot: the key's two words (hi, lo)
        std::vector<int32_t> slot_rows; // per slot: the position in its table
        bool too_many_rows = false;  // a table beyond INT32_MAX entries (positions are int32)
    };

    // One DFS per tree over every root-to-leaf path, collecting each interaction of orders
    // 2..max_order whose feature ids all occur on the path. An interaction is emitted at the
    // first occurrence of its deepest feature. Interactions are kept as 128-bit keys while they
    // fit, else as their feature ids.
    class SubsetCollector
    {
    public:
        // `max_width` is the largest number of distinct features on any root-to-leaf path; the
        // keys only have to cover min(max_order, max_width) feature ids.
        SubsetCollector(int64_t n_features, int max_order, int max_width)
            : n_features_(n_features), max_order_(max_order),
              keys_fit_(keys_fit(n_features, std::min(max_order, std::max(max_width, 1)))),
              key_sets_(static_cast<size_t>(max_order) + 1),
              row_sets_(static_cast<size_t>(max_order) + 1),
              chosen_(static_cast<size_t>(std::max(max_order, 1))),
              merged_(static_cast<size_t>(std::max(max_order, 1)))
        {
        }

        // Walks one tree; node ids are relative to the given arrays, and a leaf has
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
        // the hash index. A table of order 2 or 3 holding more than `complete_above` of all
        // C(n_features, order) interactions is completed to all of them (the kernels then rank
        // instead of probing); 1.0 never completes.
        Tables finish(double complete_above = 1.0) const
        {
            Tables t;
            t.counts.assign(static_cast<size_t>(max_order_) + 1, 0);
            t.has_index = keys_fit_;
            int64_t position = 0, slot_position = 0;
            for (int order = 2; order <= max_order_; ++order)
            {
                if (keys_fit_)
                {
                    std::vector<Key128> sorted = key_sets_[order].sorted_keys();
                    if (order <= 3 && complete_above < 1.0 && binom_double(n_features_, order) <= 2147483647.0 &&
                        static_cast<double>(sorted.size()) > complete_above * binom_double(n_features_, order))
                        sorted = complete_keys(order);
                    const int64_t count = static_cast<int64_t>(sorted.size());
                    if (count > INT32_MAX)
                    {
                        t.too_many_rows = true;
                        return t;
                    }
                    t.counts[order] = count;
                    t.keys.resize(static_cast<size_t>(position + count * order));
                    const bool complete = table_is_complete(count, order, n_features_);
                    const int bits = block_bits(count, complete);
                    const int64_t size = int64_t(1) << bits;
                    const uint64_t mask = static_cast<uint64_t>(size) - 1;
                    t.slot_keys.resize(static_cast<size_t>(2 * (slot_position + size)), -1);
                    t.slot_rows.resize(static_cast<size_t>(slot_position + size), -1);
                    int64_t *block_keys = t.slot_keys.data() + 2 * slot_position;
                    int32_t *block_rows = t.slot_rows.data() + slot_position;
                    for (int64_t r = 0; r < count; ++r)
                    {
                        Key128 key = sorted[static_cast<size_t>(r)];
                        if (!complete) // a complete table has no index entries
                        {
                            uint64_t slot = key_hash(key) >> (64 - bits);
                            while (static_cast<uint64_t>(block_keys[2 * slot]) != kEmptyKey)
                                slot = (slot + 1) & mask;
                            block_keys[2 * slot] = static_cast<int64_t>(key.hi);
                            block_keys[2 * slot + 1] = static_cast<int64_t>(key.lo);
                            block_rows[slot] = static_cast<int32_t>(r);
                        }
                        int32_t *row = t.keys.data() + position + r * order;
                        for (int d = order - 1; d >= 0; --d)
                            row[d] = static_cast<int32_t>(key_divmod(key, static_cast<uint64_t>(n_features_)));
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
        static double binom_double(int64_t n, int k)
        {
            double b = 1.0;
            for (int i = 0; i < k; ++i)
                b = b * static_cast<double>(n - i) / static_cast<double>(i + 1);
            return b;
        }

        // Every interaction of `order` over [0, n_features) as keys, in lexicographic order.
        std::vector<Key128> complete_keys(int order) const
        {
            std::vector<Key128> keys;
            keys.reserve(static_cast<size_t>(binom_double(n_features_, order)));
            const uint64_t n = static_cast<uint64_t>(n_features_);
            if (order == 2)
            {
                for (uint64_t i = 0; i < n; ++i)
                    for (uint64_t j = i + 1; j < n; ++j)
                        keys.push_back(Key128{0, i * n + j});
            }
            else
            {
                for (uint64_t i = 0; i < n; ++i)
                    for (uint64_t j = i + 1; j < n; ++j)
                        for (uint64_t k = j + 1; k < n; ++k)
                            keys.push_back(key_mul_add(Key128{0, i * n + j}, n, k));
            }
            return keys;
        }

        // Emits every interaction of order - 1 among the current path features, joined with new_feature.
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
        std::vector<FlatKeySet> key_sets_;                       // per order: the interactions as keys, while the keys fit
        std::vector<std::set<std::vector<int32_t>>> row_sets_;  // per order: the interactions as feature ids, otherwise
        std::vector<int> chosen_;
        std::vector<int32_t> merged_;
    };
}  // namespace subset_tables

namespace subset_tables
{
    // Largest number of distinct split features on any root-to-leaf path of the ensemble.
    template <typename Int>
    inline int max_path_width(const Int *features, const Int *children_left, const Int *children_right,
                              const Int *tree_offsets, int64_t n_trees)
    {
        struct Frame
        {
            int64_t node;
            int depth; // path positions [0, depth) lead to this node
            int width; // distinct features among them
        };
        int best = 0;
        std::vector<int> path;
        std::vector<Frame> stack;
        for (int64_t t = 0; t < n_trees; ++t)
        {
            const Int *f = features + tree_offsets[t];
            const Int *cl = children_left + tree_offsets[t];
            const Int *cr = children_right + tree_offsets[t];
            stack.clear();
            stack.push_back(Frame{0, 0, 0});
            while (!stack.empty())
            {
                const Frame frame = stack.back();
                stack.pop_back();
                if (cl[frame.node] == cr[frame.node])
                {
                    best = std::max(best, frame.width);
                    continue;
                }
                const int feature = static_cast<int>(f[frame.node]);
                bool seen = false;
                for (int d = 0; d < frame.depth && !seen; ++d)
                    seen = path[static_cast<size_t>(d)] == feature;
                path.resize(static_cast<size_t>(frame.depth) + 1);
                path[static_cast<size_t>(frame.depth)] = feature;
                const int width = frame.width + (seen ? 0 : 1);
                stack.push_back(Frame{static_cast<int64_t>(cl[frame.node]), frame.depth + 1, width});
                stack.push_back(Frame{static_cast<int64_t>(cr[frame.node]), frame.depth + 1, width});
            }
        }
        return best;
    }
} // namespace subset_tables

#ifdef Py_PYTHON_H
// === preprocess_subset_tables (Python binding) ===
// preprocess_subset_tables(features, children_left, children_right, tree_offsets, n_features,
//                          max_order, complete_above=1.0)
//   -> ((keys, counts), (slot_keys, slot_rows) or None)
// Node arrays: the trees back to back with tree-relative child ids, tree t owning nodes
//   [tree_offsets[t], tree_offsets[t + 1]); a leaf has children_left == children_right.
// keys: the tables of orders 2..max_order back to back, each entry its ascending feature ids.
// counts: entries per order (max_order + 1 entries).
// slot_keys, slot_rows: the hash index, two int64 words (hi, lo) and one position per slot;
//   None when a key of the widest path would overflow 127 bits (the kernels then search).
// complete_above < 1.0 completes an order-2/3 table holding more than that fraction of all
//   C(n_features, order) interactions to all of them.
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

    // Compares the array lengths a kernel received with the geometry derived from `counts`;
    // returns a message or NULL. `n_slots` is -1 without an index, `n_outputs` -1 when unchecked.
    inline const char *check_layout(const Geometry &g, int64_t n_keys, int64_t n_slots, int64_t n_outputs)
    {
        if (n_keys != g.n_keys)
            return "the interaction tables are inconsistent with the keys length.";
        if (n_slots >= 0 && n_slots != g.n_slots)
            return "the index arrays do not match the block sizes derived from counts.";
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
    double complete_above = 1.0;
    if (!PyArg_ParseTuple(args, "OOOOii|d", &features_obj, &children_left_obj, &children_right_obj,
                          &tree_offsets_obj, &n_features, &max_order, &complete_above))
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

    const int max_width = subset_tables::max_path_width(features, cl, cr, toff, n_trees);
    subset_tables::SubsetCollector collector(n_features, max_order, max_width);
    for (int64_t t = 0; t < n_trees; ++t)
        collector.walk(features + toff[t], cl + toff[t], cr + toff[t], 0);
    release();
    const subset_tables::Tables tables = collector.finish(complete_above);
    if (tables.too_many_rows)
    {
        PyErr_SetString(PyExc_ValueError, "more than 2^31 interactions of one order are not supported.");
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
// Reads a kernel's output array back into a dict: the order-1 block by feature id, then every
// entry of every table. feature_ids: int64 array of n_features entries renaming the kernel's
// feature ids, or None to keep them. skip_zeros: leave out entries that are exactly zero.
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
            arg_error = "interaction counts must be non-negative.";
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
