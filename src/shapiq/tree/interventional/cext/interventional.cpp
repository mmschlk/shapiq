#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include "weights.cpp"
#include "utils.cpp"

namespace algorithms
{

    using SparseInteractionMap = std::unordered_map<BitSet, double, BitSetHash, BitSetEqual>;

    // Multiplier of the multiply-shift subset hash; must equal INDEX_HASH_MULTIPLIER in
    // shapiq/tree/subset_index.py (2^64 / golden ratio).
    constexpr uint64_t kIndexHashMultiplier = 0x9E3779B97F4A7C15ULL;

    // A feature subset as one int64: its sorted features read as the digits (feature + 1) of a
    // number in base num_features + 1. The +1 keeps a leading feature 0 from vanishing, so
    // subsets of different sizes never share a key. Usable while
    // (num_features + 1)^max_order fits in an int64 (see flat_subset_keys_fit).
    inline bool flat_subset_keys_fit(int num_features, int max_order)
    {
        const int64_t base = static_cast<int64_t>(num_features) + 1;
        int64_t cap = 1;
        for (int i = 0; i < max_order; ++i)
        {
            if (cap > INT64_MAX / base)
                return false;
            cap *= base;
        }
        return max_order < 64;  // the enumeration keeps at most 63 chosen features
    }

    // Open-addressing map from a subset key to its accumulated contribution, filled per
    // explanation with exactly the subsets the E/R sets of its leaves produce. Replaces the
    // std::unordered_map<BitSet, double> of the sparse path: no allocation per entry, an
    // integer key, a multiply-shift hash (the multiplier of the quadrature subset index), and
    // linear probing in contiguous arrays. Grows by doubling to stay at most half full.
    class FlatSubsetMap
    {
    public:
        explicit FlatSubsetMap(size_t initial_capacity = 1024) { reset(initial_capacity); }

        void add(int64_t key, double value)
        {
            uint64_t slot = home(key);
            while (true)
            {
                const int64_t stored = keys_[slot];
                if (stored == key)
                {
                    values_[slot] += value;
                    return;
                }
                if (stored == kEmpty)
                {
                    if (2 * (size_ + 1) > keys_.size())
                    {
                        grow();
                        add(key, value);
                        return;
                    }
                    keys_[slot] = key;
                    values_[slot] = value;
                    ++size_;
                    return;
                }
                slot = (slot + 1) & mask_;
            }
        }

        void merge_from(const FlatSubsetMap &other)
        {
            other.for_each([&](int64_t key, double value) { add(key, value); });
        }

        template <typename Func>
        void for_each(Func &&f) const
        {
            for (size_t slot = 0; slot < keys_.size(); ++slot)
            {
                if (keys_[slot] != kEmpty)
                    f(keys_[slot], values_[slot]);
            }
        }

        size_t size() const { return size_; }

    private:
        static constexpr int64_t kEmpty = -1;

        void reset(size_t capacity)
        {
            int bits = 1;
            while ((size_t(1) << bits) < capacity)
                ++bits;
            keys_.assign(size_t(1) << bits, kEmpty);
            values_.assign(size_t(1) << bits, 0.0);
            mask_ = (uint64_t(1) << bits) - 1;
            shift_ = 64 - bits;
            size_ = 0;
        }

        uint64_t home(int64_t key) const
        {
            return (static_cast<uint64_t>(key) * kIndexHashMultiplier) >> shift_;
        }

        void grow()
        {
            std::vector<int64_t> old_keys = std::move(keys_);
            std::vector<double> old_values = std::move(values_);
            reset(old_keys.size() * 2);
            for (size_t slot = 0; slot < old_keys.size(); ++slot)
            {
                if (old_keys[slot] != kEmpty)
                    add(old_keys[slot], old_values[slot]);
            }
        }

        std::vector<int64_t> keys_;
        std::vector<double> values_;
        uint64_t mask_ = 0;
        int shift_ = 63;
        size_t size_ = 0;
    };

    inline int get_interaction_index(int i, int j, int num_features, int max_order)
    {
        // Helper function to compute compact index for order-2 interactions
        // For max_order=1: just returns feature (linear indexing)
        // For max_order=2: maps (i,j) pairs to compact indices:
        //   - Indices 0..n-1: main effects (i,i)
        //   - Indices n onwards: pairwise (i,j) where i < j in row-major upper triangle order
        if (max_order > 3)
        {
            throw std::invalid_argument("get_interaction_index only supports max_order 1, 2, or 3");
        }
        if (max_order == 1)
        {
            return i;
        }
        if (i == j)
        {
            return i;
        }
        if (i > j)
        {
            std::swap(i, j);
        }
        // We store as upper triangle without diagonal (since diagonal are the main effects).
        // The first num_features are the main effects (i,i).
        // The first term shifts us past the main effects.
        // Then we calculate the offset for the upper triangle index.
        // The number of interactions before row i is the sum of (num_features - 1) + (num_features - 2) + ... + (num_features - i) = i * num_features - i*(i+1)/2.
        // Then we add (j - i - 1) to get to the correct column within row i.
        return num_features + (i * num_features - i * (i + 1) / 2) + (j - i - 1);
    }

    inline int get_interaction_index3(int i, int j, int k, int num_features)
    {
        // Ensure i < j < k
        if (i > j) std::swap(i, j);
        if (j > k) std::swap(j, k);
        if (i > j) std::swap(i, j);
        int base = num_features + num_features * (num_features - 1) / 2;
        return base + i + j * (j - 1) / 2 + k * (k - 1) * (k - 2) / 6;
    }

    void enumerate_r_subsets(const uint64_t *r_features,
                             int r_count,
                             int start_idx,
                             int remaining,
                             BitSet &subset,
                             double contribution,
                             SparseInteractionMap &interactions)
    {
        if (remaining == 0)
        {
            interactions[subset] += contribution;
            return;
        }

        const int n = r_count;
        for (int i = start_idx; i <= n - remaining; ++i)
        {
            const int64_t feature_id = static_cast<int64_t>(r_features[static_cast<size_t>(i)]);
            subset.add(feature_id);
            enumerate_r_subsets(r_features, r_count, i + 1, remaining - 1, subset, contribution, interactions);
            subset.remove(feature_id);
        }
    }

    void enumerate_e_subsets(const uint64_t *e_features,
                             int e_count,
                             const uint64_t *r_features,
                             int r_count,
                             int e_start_idx,
                             int e_remaining,
                             int r_remaining,
                             BitSet &subset,
                             double contribution,
                             SparseInteractionMap &interactions)
    {
        if (e_remaining == 0)
        {
            enumerate_r_subsets(r_features, r_count, 0, r_remaining, subset, contribution, interactions);
            return;
        }

        const int n = e_count;
        for (int i = e_start_idx; i <= n - e_remaining; ++i)
        {
            const int64_t feature_id = static_cast<int64_t>(e_features[static_cast<size_t>(i)]);
            subset.add(feature_id);
            enumerate_e_subsets(e_features, e_count, r_features, r_count, i + 1, e_remaining - 1, r_remaining, subset, contribution, interactions);
            subset.remove(feature_id);
        }
    }

    // State of one flat enumeration: s_e features chosen from E and s_r from R (each list and
    // each choice ascending), merged into one sorted subset and keyed at the leaf.
    struct FlatEnumeration
    {
        const uint64_t *e_features;
        int e_count;
        const uint64_t *r_features;
        int r_count;
        int64_t key_base;  // num_features + 1
        FlatSubsetMap *interactions;
        int s_e = 0;
        int s_r = 0;
        double contribution = 0.0;
        uint64_t chosen_e[64];
        uint64_t chosen_r[64];
    };

    inline void emit_flat_subset(FlatEnumeration &en)
    {
        // E and R are disjoint, so merging the two ascending choices gives the sorted subset
        int64_t key = 0;
        int i = 0, j = 0;
        while (i < en.s_e || j < en.s_r)
        {
            uint64_t feature;
            if (j == en.s_r || (i < en.s_e && en.chosen_e[i] < en.chosen_r[j]))
                feature = en.chosen_e[i++];
            else
                feature = en.chosen_r[j++];
            key = key * en.key_base + static_cast<int64_t>(feature) + 1;
        }
        en.interactions->add(key, en.contribution);
    }

    // Same subsets in the same order as enumerate_r_subsets / enumerate_e_subsets, so each
    // subset receives its contributions in the same sequence as in the BitSet-keyed map.
    void enumerate_r_flat(FlatEnumeration &en, int start, int depth)
    {
        if (depth == en.s_r)
        {
            emit_flat_subset(en);
            return;
        }
        for (int i = start; i <= en.r_count - (en.s_r - depth); ++i)
        {
            en.chosen_r[depth] = en.r_features[i];
            enumerate_r_flat(en, i + 1, depth + 1);
        }
    }

    void enumerate_e_flat(FlatEnumeration &en, int start, int depth)
    {
        if (depth == en.s_e)
        {
            enumerate_r_flat(en, 0, 0);
            return;
        }
        for (int i = start; i <= en.e_count - (en.s_e - depth); ++i)
        {
            en.chosen_e[depth] = en.e_features[i];
            enumerate_e_flat(en, i + 1, depth + 1);
        }
    }

    void sparse_order_update(const StackFrame &frame,
                             double value,
                             SparseInteractionMap &interactions,
                             inter_weights::WeightCache &weight_cache,
                             int num_features,
                             IndexType index,
                             int max_order,
                             uint64_t *e_buffer,
                             uint64_t *r_buffer,
                             uint64_t e_count,
                             uint64_t r_count)
    {
        if (e_count > 0)
        {
            frame.E.fill_buffer(e_buffer);
        }
        if (r_count > 0)
        {
            frame.R.fill_buffer(r_buffer);
        }

        BitSet subset(num_features);
        // Compute contributions for all subsets of E and R up to the specified max_order.
        //  We iterate over all possible subset sizes s from 1 to max_order.
        // For each subset size s, we determine how many features in the subset come from E (s_cap_e) and how many come from R (s_cap_r).
        // We then compute the weight for that combination of features using the weight cache and call the recursive enumeration functions to generate all subsets of E and R with the specified number of features, updating the interactions map with the contributions.
        for (int s = 1; s <= max_order; ++s)
        {
            int min_from_e = std::max(0, s - static_cast<int>(r_count));
            int max_from_e = std::min(s, static_cast<int>(e_count));
            for (int s_cap_e = min_from_e; s_cap_e <= max_from_e; ++s_cap_e)
            {
                int s_cap_r = s - s_cap_e;
                const double weight = weight_cache.get_weight(num_features, frame.e, frame.r, s_cap_e, s_cap_r, s, index, max_order);
                if (weight == 0.0)
                {
                    continue;
                }
                const double contribution = static_cast<double>(value) * weight;
                // We now update all the interactions corresponding to subsets of E and R with s_cap_e features from E and s_cap_r features from R by calling the enumerate_e_subsets function, which will recursively generate all such subsets and update the interactions map with the computed contribution for each subset.
                enumerate_e_subsets(
                    e_buffer,
                    static_cast<int>(e_count),
                    r_buffer,
                    static_cast<int>(r_count),
                    0,
                    s_cap_e,
                    s_cap_r,
                    subset,
                    contribution,
                    interactions);
            }
        }
    }

    // Walks one tree for the explain point against one reference sample and hands every leaf
    // reached to `leaf_update`: its frame (E and R) plus scratch buffers sized for E and R.
    // `stack` is the caller's per-thread scratch, reused across calls: allocating a fresh one
    // per (tree, reference) pair made threads contend in the allocator.
    template <typename LeafUpdate>
    void traverse_explain_vs_reference(Tree &tree,
                                       double *reference_data,
                                       double *explain_data,
                                       int num_features,
                                       std::vector<StackFrame> &stack,
                                       LeafUpdate &&leaf_update)
    {
        stack.clear();
        BitSet empty_A(num_features);
        BitSet empty_B(num_features);

        uint64_t bit_buffer_E[64];
        uint64_t bit_buffer_R[64];
        std::vector<uint64_t> vector_buffer_E;
        std::vector<uint64_t> vector_buffer_R;
        uint64_t *e_buffer;
        uint64_t *r_buffer;
        uint64_t e_count;
        uint64_t r_count;

        stack.push_back(StackFrame(0, empty_A, empty_B, 0, 0));
        while (!stack.empty())
        {
            StackFrame current_frame = std::move(stack.back());
            stack.pop_back();
            int64_t node_id = current_frame.node_id;
            const BitSet &E = current_frame.E;
            const BitSet &R = current_frame.R;
            uint64_t e = current_frame.e;
            uint64_t r = current_frame.r;
            e_count = E.num_bits();
            r_count = R.num_bits();

            bool is_leaf = tree.is_leaf(node_id);

            if (!is_leaf)
            {
                int64_t feature_id = tree.features[node_id];
                int64_t child_explain_point = tree.goes_left(explain_data[feature_id], node_id) ? tree.children_left[node_id] : tree.children_right[node_id];
                int64_t child_reference_point = tree.goes_left(reference_data[feature_id], node_id) ? tree.children_left[node_id] : tree.children_right[node_id];

                if (child_explain_point == child_reference_point)
                {
                    stack.push_back(StackFrame(child_explain_point, E, R, e, r));
                }
                else
                {
                    if (!R.contains(feature_id))
                    {
                        BitSet next_E = E;
                        bool added_to_E = next_E.add(feature_id);
                        stack.push_back(StackFrame(child_explain_point, next_E, R, e + (added_to_E ? 1 : 0), r));
                    }
                    if (!E.contains(feature_id))
                    {
                        BitSet next_R = R;
                        bool added_to_R = next_R.add(feature_id);
                        stack.push_back(StackFrame(child_reference_point, E, next_R, e, r + (added_to_R ? 1 : 0)));
                    }
                }
            }
            else
            {
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
                double leaf_value = tree.leaf_predictions[node_id];
                leaf_update(current_frame, leaf_value, e_buffer, r_buffer, e_count, r_count);
            }
        }
    }

    void compute_interactions_sparse(Tree tree,
                                      SparseInteractionMap &interactions,
                                      inter_weights::WeightCache &weight_cache,
                                      double *reference_data,
                                      double *explain_data,
                                      int num_features,
                                      IndexType index,
                                      int max_order,
                                      int verbose,
                                      std::vector<StackFrame> &stack)
    {
        traverse_explain_vs_reference(
            tree, reference_data, explain_data, num_features, stack,
            [&](const StackFrame &frame, double leaf_value, uint64_t *e_buffer, uint64_t *r_buffer,
                uint64_t e_count, uint64_t r_count)
            {
                sparse_order_update(frame, leaf_value, interactions, weight_cache, num_features, index,
                                    max_order, e_buffer, r_buffer, e_count, r_count);
            });
    }

    // The sparse path on a FlatSubsetMap (see flat_subset_keys_fit for when keys fit).
    void compute_interactions_flat(Tree tree,
                                   FlatSubsetMap &interactions,
                                   inter_weights::WeightCache &weight_cache,
                                   double *reference_data,
                                   double *explain_data,
                                   int num_features,
                                   IndexType index,
                                   int max_order,
                                   std::vector<StackFrame> &stack)
    {
        FlatEnumeration en;
        en.key_base = static_cast<int64_t>(num_features) + 1;
        en.interactions = &interactions;
        traverse_explain_vs_reference(
            tree, reference_data, explain_data, num_features, stack,
            [&](const StackFrame &frame, double leaf_value, uint64_t *e_buffer, uint64_t *r_buffer,
                uint64_t e_count, uint64_t r_count)
            {
                if (e_count > 0)
                    frame.E.fill_buffer(e_buffer);
                if (r_count > 0)
                    frame.R.fill_buffer(r_buffer);
                en.e_features = e_buffer;
                en.e_count = static_cast<int>(e_count);
                en.r_features = r_buffer;
                en.r_count = static_cast<int>(r_count);
                // same sizes, splits and weights as sparse_order_update
                for (int s = 1; s <= max_order; ++s)
                {
                    int min_from_e = std::max(0, s - static_cast<int>(r_count));
                    int max_from_e = std::min(s, static_cast<int>(e_count));
                    for (int s_cap_e = min_from_e; s_cap_e <= max_from_e; ++s_cap_e)
                    {
                        int s_cap_r = s - s_cap_e;
                        const double weight = weight_cache.get_weight(num_features, frame.e, frame.r, s_cap_e, s_cap_r, s, index, max_order);
                        if (weight == 0.0)
                            continue;
                        en.contribution = static_cast<double>(leaf_value) * weight;
                        en.s_e = s_cap_e;
                        en.s_r = s_cap_r;
                        enumerate_e_flat(en, 0, 0);
                    }
                }
            });
    }


    void first_order_bitset_update(const BitSet E, const BitSet R, double value, double *interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index)
    {
        const double wE = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 1, 0, 1, index, 1));
        const double wR = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 1, 1, index, 1));
        E.for_each_set_bit([&](uint64_t feature)
                                         { interactions[feature] += value * wE; });
        R.for_each_set_bit([&](uint64_t feature)
                                         { interactions[feature] += value * wR; });
    }

    void second_order_bitset_update(const BitSet E, const BitSet R,
        uint64_t *e_buffer, uint64_t *r_buffer,
        uint64_t e_count, uint64_t r_count,
        double value, double *interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index)
    {
        // Main Effect updates (diagonal elements)
        const double wE = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(),R.num_bits(), 1, 0, 1, index, 2));
        const double wR = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 1, 1, index, 2));
        E.for_each_set_bit([&](uint64_t feature)
                                 {
            int idx = get_interaction_index(static_cast<int>(feature), static_cast<int>(feature), num_features, 2);
            interactions[idx] += value * wE; });
        R.for_each_set_bit([&](uint64_t feature)
                                 {
            int idx = get_interaction_index(static_cast<int>(feature), static_cast<int>(feature), num_features, 2);
            interactions[idx] += value * wR; });

        // Fill the buffers with the feature indices in E and R.
        // This avoids repeated memory allocations during the interaction updates.
        E.fill_buffer(e_buffer);
        R.fill_buffer(r_buffer);

        // Interaction in E (upper triangle)
        const double wEE = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 2, 0, 2, index, 2));
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = i + 1; j < e_count; j++)
            {
                int idx = get_interaction_index(static_cast<int>(e_buffer[i]), static_cast<int>(e_buffer[j]), num_features, 2);
                interactions[idx] += value * wEE;
            }
        }
        // Interactions in R (upper triangle)
        const double wRR = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 2, 2, index, 2));
        for (size_t i = 0; i < r_count; i++)
        {
            for (size_t j = i + 1; j < r_count; j++)
            {
                int idx = get_interaction_index(static_cast<int>(r_buffer[i]), static_cast<int>(r_buffer[j]), num_features, 2);
                interactions[idx] += value * wRR;
            }
        }
        // Cross interactions (E × R)
        const double wER = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 1, 1, 2, index, 2));
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = 0; j < r_count; j++)
            {
                if (e_buffer[i] == r_buffer[j])
                    continue; // Skip if the same feature is in both E and R, as this would correspond to a main effect, not an interaction.
                int idx = get_interaction_index(static_cast<int>(e_buffer[i]), static_cast<int>(r_buffer[j]), num_features, 2);
                interactions[idx] += value * wER;
            }
        }
    }

    void third_order_bitset_update(const BitSet E, const BitSet R,
        uint64_t *e_buffer, uint64_t *r_buffer,
        uint64_t e_count, uint64_t r_count,
        double value, double *interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index, int max_order, int verbose)
    {
        // Precompute all weight types
        const double wE   = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 1, 0, 1, index, max_order));
        const double wR   = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 1, 1, index, max_order));
        const double wEE  = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 2, 0, 2, index, max_order));
        const double wRR  = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 2, 2, index, max_order));
        const double wER  = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 1, 1, 2, index, max_order));
        const double wEEE = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 3, 0, 3, index, max_order));
        const double wRRR = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 3, 3, index, max_order));
        const double wEER = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 2, 1, 3, index, max_order));
        const double wERR = static_cast<double>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 1, 2, 3, index, max_order));

        E.fill_buffer(e_buffer);
        R.fill_buffer(r_buffer);

        // Order-1
        for (size_t i = 0; i < e_count; i++)
        {
            int idx = get_interaction_index(static_cast<int>(e_buffer[i]), static_cast<int>(e_buffer[i]), num_features, 2);
            interactions[idx] += value * wE;
        }
        for (size_t i = 0; i < r_count; i++)
        {
            int idx = get_interaction_index(static_cast<int>(r_buffer[i]), static_cast<int>(r_buffer[i]), num_features, 2);
            interactions[idx] += value * wR;
        }

        // Order-2 EE
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = i + 1; j < e_count; j++)
            {
                int idx = get_interaction_index(static_cast<int>(e_buffer[i]), static_cast<int>(e_buffer[j]), num_features, 2);
                interactions[idx] += value * wEE;
            }
        }
        // Order-2 RR
        for (size_t i = 0; i < r_count; i++)
        {
            for (size_t j = i + 1; j < r_count; j++)
            {
                int idx = get_interaction_index(static_cast<int>(r_buffer[i]), static_cast<int>(r_buffer[j]), num_features, 2);
                interactions[idx] += value * wRR;
            }
        }
        // Order-2 ER
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = 0; j < r_count; j++)
            {
                if (e_buffer[i] == r_buffer[j]) continue;
                int idx = get_interaction_index(static_cast<int>(e_buffer[i]), static_cast<int>(r_buffer[j]), num_features, 2);
                interactions[idx] += value * wER;
            }
        }

        // Order-3 EEE
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = i + 1; j < e_count; j++)
            {
                for (size_t k = j + 1; k < e_count; k++)
                {
                    int idx = get_interaction_index3(static_cast<int>(e_buffer[i]), static_cast<int>(e_buffer[j]), static_cast<int>(e_buffer[k]), num_features);
                    interactions[idx] += value * wEEE;
                }
            }
        }
        // Order-3 RRR
        for (size_t i = 0; i < r_count; i++)
        {
            for (size_t j = i + 1; j < r_count; j++)
            {
                for (size_t k = j + 1; k < r_count; k++)
                {
                    int idx = get_interaction_index3(static_cast<int>(r_buffer[i]), static_cast<int>(r_buffer[j]), static_cast<int>(r_buffer[k]), num_features);
                    interactions[idx] += value * wRRR;
                }
            }
        }
        // Order-3 EER
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = i + 1; j < e_count; j++)
            {
                for (size_t k = 0; k < r_count; k++)
                {
                    uint64_t rk = r_buffer[k];
                    if (e_buffer[i] == rk || e_buffer[j] == rk) continue;
                    int idx = get_interaction_index3(static_cast<int>(e_buffer[i]), static_cast<int>(e_buffer[j]), static_cast<int>(rk), num_features);
                    interactions[idx] += value * wEER;
                }
            }
        }
        // Order-3 ERR
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = 0; j < r_count; j++)
            {
                if (e_buffer[i] == r_buffer[j]) continue;
                for (size_t k = j + 1; k < r_count; k++)
                {
                    if (e_buffer[i] == r_buffer[k]) continue;
                    int idx = get_interaction_index3(static_cast<int>(e_buffer[i]), static_cast<int>(r_buffer[j]), static_cast<int>(r_buffer[k]), num_features);
                    interactions[idx] += value * wERR;
                }
            }
        }
    }

    void any_order_bitset_update(const BitSet E, const BitSet R,
        uint64_t *e_buffer, uint64_t *r_buffer,
        uint64_t e_count, uint64_t r_count,
        double value, SparseInteractionMap &interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index, int max_order, int verbose)
    {

        if (e_count > 0)
        {
            E.fill_buffer(e_buffer);
        }
        if (r_count > 0)
        {
            R.fill_buffer(r_buffer);
        }

        BitSet subset(num_features);
        // Compute contributions for all subsets of E and R up to the specified max_order.
        //  We iterate over all possible subset sizes s from 1 to max_order.
        // For each subset size s, we determine how many features in the subset come from E (s_cap_e) and how many come from R (s_cap_r).
        // We then compute the weight for that combination of features using the weight cache and call the recursive enumeration functions to generate all subsets of E and R with the specified number of features, updating the interactions map with the contributions.
        for (int s = 1; s <= max_order; ++s)
        {
            int min_from_e = std::max(0, s - static_cast<int>(r_count));
            int max_from_e = std::min(s, static_cast<int>(e_count));
            for (int s_cap_e = min_from_e; s_cap_e <= max_from_e; ++s_cap_e)
            {
                int s_cap_r = s - s_cap_e;
                const double weight = weight_cache.get_weight(num_features, e_count, r_count, s_cap_e, s_cap_r, s, index, max_order);
                if (weight == 0.0)
                {
                    continue;
                }
                const double contribution = static_cast<double>(value) * weight;
                // We now update all the interactions corresponding to subsets of E and R with s_cap_e features from E and s_cap_r features from R by calling the enumerate_e_subsets function, which will recursively generate all such subsets and update the interactions map with the computed contribution for each subset.
                enumerate_e_subsets(
                    e_buffer,
                    static_cast<int>(e_count),
                    r_buffer,
                    static_cast<int>(r_count),
                    0,
                    s_cap_e,
                    s_cap_r,
                    subset,
                    contribution,
                    interactions);
            }
        }
    }



}
