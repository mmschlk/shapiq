#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include "weights.cpp"
#include "utils.cpp"
#include "../../cext/subset_tables.hpp"

namespace algorithms
{

    using SparseInteractionMap = std::unordered_map<BitSet, double, BitSetHash, BitSetEqual>;

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
    // integer key, the multiply-shift hash of the shared subset index, and
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
            return (static_cast<uint64_t>(key) * subset_tables::kIndexHashMultiplier) >> shift_;
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

}
