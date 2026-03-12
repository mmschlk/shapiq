#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <tuple>
#include <algorithm>
#include "weights.cpp"
#include "utils.cpp"

namespace algorithms
{

    using SparseInteractionMap = std::unordered_map<BitSet, double, BitSetHash, BitSetEqual>;

    inline int get_interaction_index(int i, int j, int num_features, int max_order)
    {
        // Helper function to compute compact index for order-2 interactions
        // For max_order=1: just returns feature (linear indexing)
        // For max_order=2: maps (i,j) pairs to compact indices:
        //   - Indices 0..n-1: main effects (i,i)
        //   - Indices n onwards: pairwise (i,j) where i < j in row-major upper triangle order
        if (max_order > 2)
        {
            throw std::invalid_argument("get_interaction_index only supports max_order 1 or 2");
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

    void first_order_update(const StackFrame &frame, float value, double *interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index, int max_order, int verbose)
    {
        const float wE = static_cast<float>(weight_cache.get_weight(num_features, frame.e, frame.r, 1, 0, 1, index, max_order));
        const float wR = static_cast<float>(weight_cache.get_weight(num_features, frame.e, frame.r, 0, 1, 1, index, max_order));

        frame.E.for_each_set_bit([&](uint64_t feature)
                                         { interactions[feature] += value * wE; });
        frame.R.for_each_set_bit([&](uint64_t feature)
                                         { interactions[feature] += value * wR; });
    }

    void second_order_update(const StackFrame &frame,
                             uint64_t *e_buffer, uint64_t *r_buffer,
                             uint64_t e_count, uint64_t r_count,
                             float value, double *interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index, int max_order, int verbose)
    {

        // Main Effect updates (diagonal elements)
        const float wE = static_cast<float>(weight_cache.get_weight(num_features, frame.e, frame.r, 1, 0, 1, index, max_order));
        const float wR = static_cast<float>(weight_cache.get_weight(num_features, frame.e, frame.r, 0, 1, 1, index, max_order));
        frame.E.for_each_set_bit([&](uint64_t feature)
                                 {
            int idx = get_interaction_index(static_cast<int>(feature), static_cast<int>(feature), num_features, max_order);
            interactions[idx] += value * wE; });
        frame.R.for_each_set_bit([&](uint64_t feature)
                                 {
            int idx = get_interaction_index(static_cast<int>(feature), static_cast<int>(feature), num_features, max_order);
            interactions[idx] += value * wR; });

        // Fill the buffers with the feature indices in E and R.
        // This avoids repeated memory allocations during the interaction updates.
        frame.E.fill_buffer(e_buffer);
        frame.R.fill_buffer(r_buffer);

        // Interaction in E (upper triangle)
        const float wEE = static_cast<float>(weight_cache.get_weight(num_features, frame.e, frame.r, 2, 0, 2, index, max_order));
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = i + 1; j < e_count; j++)
            {
                int idx = get_interaction_index(static_cast<int>(e_buffer[i]), static_cast<int>(e_buffer[j]), num_features, max_order);
                interactions[idx] += value * wEE;
            }
        }
        // Interactions in R (upper triangle)
        const float wRR = static_cast<float>(weight_cache.get_weight(num_features, frame.e, frame.r, 0, 2, 2, index, max_order));
        for (size_t i = 0; i < r_count; i++)
        {
            for (size_t j = i + 1; j < r_count; j++)
            {
                int idx = get_interaction_index(static_cast<int>(r_buffer[i]), static_cast<int>(r_buffer[j]), num_features, max_order);
                interactions[idx] += value * wRR;
            }
        }
        // Cross interactions (E × R)
        const float wER = static_cast<float>(weight_cache.get_weight(num_features, frame.e, frame.r, 1, 1, 2, index, max_order));
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = 0; j < r_count; j++)
            {
                if (e_buffer[i] == r_buffer[j])
                    continue; // Skip if the same feature is in both E and R, as this would correspond to a main effect, not an interaction.
                int idx = get_interaction_index(static_cast<int>(e_buffer[i]), static_cast<int>(r_buffer[j]), num_features, max_order);
                interactions[idx] += value * wER;
            }
        }
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

    void sparse_order_update(const StackFrame &frame,
                             float value,
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

    void compute_interactions(Tree tree, double *interactions,
                               inter_weights::WeightCache &weight_cache,
                               float *reference_data,
                               float *explain_data,
                               int num_features,
                               IndexType index,
                               int max_order,
                               int verbose)
    {
        /**
         * This function computes the first-order interactions for a given decision tree and input data.
         * The algorithm computes the index based on the algorithm of Zern. 2023 (Algorithm 1) "Interventional SHAP Values and Interaction Values for Piecewise Linear Regression Trees "
         * The output is an array of interaction values for each feature.
         * This algorithm here keep storage of [A, N\B] pairs on the stack, where A and B are the sets of features which are necessary to reach the current node based on the explain and reference point, respectively.
         */

        std::vector<StackFrame> stack;
        BitSet empty_A(num_features);
        BitSet empty_B(num_features);

        // Create buffers for the feature indices in E and R to avoid repeated memory allocations during the interaction updates.
        uint64_t bit_buffer_E[64];
        uint64_t bit_buffer_R[64];
        std::vector<uint64_t> vector_buffer_E;
        std::vector<uint64_t> vector_buffer_R;

        uint64_t *e_buffer;
        uint64_t *r_buffer;
        uint64_t e_count;
        uint64_t r_count;

        stack.reserve(1000);                                    // Reserve space for 1000 stack frames to avoid frequent reallocations. Adjust this number based on expected tree depth and branching factor.
        stack.push_back(StackFrame(0, empty_A, empty_B, 0, 0)); // Start with the root node (node_id = 0) and empty sets A and B
        while (!stack.empty())
        {
            // std::move is used to efficiently transfer only the pointer of the memory allocated for the StackFrame on the heap instead of copying the entire StackFrame, which can be expensive if it contains large data structures.
            // After std::move, the current_frame variable takes ownership of the StackFrame object that was previously owned by the last element of the stack vector.
            // This allows us to pop the last element from the stack without having to copy its contents, improving performance.
            StackFrame current_frame = std::move(stack.back()); // Using std::move to avoid unnecessary copying of the StackFrame when popping from the stack.
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
                // 1. Case: Both points go to the same child node
                if (child_explain_point == child_reference_point)
                {
                    stack.push_back(StackFrame(child_explain_point, E, R, e, r));
                }
                else
                {
                    // 2. Case: Points go to different child nodes
                    //  Add feature_id to E iff. i not contained in R
                    if (!R.contains(feature_id))
                    {
                        BitSet next_E = E;
                        bool added_to_E = next_E.add(feature_id);
                        stack.push_back(StackFrame(child_explain_point, next_E, R, e + (added_to_E ? 1 : 0), r));
                    }
                    // Add feature_id to B iff. i not contained in A
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
                // Get the correct buffer for E and R based on their sizes. For small sets (up to 64 features), we can use the pre-allocated bit buffers.
                // For larger sets, we need to resize the vector buffers and use their data pointers.
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
                float leaf_value = tree.leaf_predictions[node_id];
                if (max_order == 1)
                    algorithms::first_order_update(current_frame, leaf_value, interactions, weight_cache, num_features, index, max_order, verbose);
                else if (max_order == 2)
                    algorithms::second_order_update(current_frame, e_buffer, r_buffer, e_count, r_count, leaf_value, interactions, weight_cache, num_features, index, max_order, verbose);
                else
                    throw std::invalid_argument("Unsupported max_order: " + std::to_string(max_order));
            }
        }
    }

    void compute_interactions_sparse(Tree tree,
                                      SparseInteractionMap &interactions,
                                      inter_weights::WeightCache &weight_cache,
                                      float *reference_data,
                                      float *explain_data,
                                      int num_features,
                                      IndexType index,
                                      int max_order,
                                      int verbose)
    {
        std::vector<StackFrame> stack;
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

        stack.reserve(1000);
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
                float leaf_value = tree.leaf_predictions[node_id];
                sparse_order_update(current_frame, leaf_value, interactions, weight_cache, num_features, index, max_order, e_buffer, r_buffer, e_count, r_count);
            }
        }
    }


    void first_order_bitset_update(const BitSet E, const BitSet R, float value, double *interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index)
    {
        const float wE = static_cast<float>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 1, 0, 1, index, 1));
        const float wR = static_cast<float>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 1, 1, index, 1));
        E.for_each_set_bit([&](uint64_t feature)
                                         { interactions[feature] += value * wE; });
        R.for_each_set_bit([&](uint64_t feature)
                                         { interactions[feature] += value * wR; });
    }

    void second_order_bitset_update(const BitSet E, const BitSet R,
        uint64_t *e_buffer, uint64_t *r_buffer,
        uint64_t e_count, uint64_t r_count,
        float value, double *interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index)
    {
        // Main Effect updates (diagonal elements)
        const float wE = static_cast<float>(weight_cache.get_weight(num_features, E.num_bits(),R.num_bits(), 1, 0, 1, index, 2));
        const float wR = static_cast<float>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 1, 1, index, 2));
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
        const float wEE = static_cast<float>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 2, 0, 2, index, 2));
        for (size_t i = 0; i < e_count; i++)
        {
            for (size_t j = i + 1; j < e_count; j++)
            {
                int idx = get_interaction_index(static_cast<int>(e_buffer[i]), static_cast<int>(e_buffer[j]), num_features, 2);
                interactions[idx] += value * wEE;
            }
        }
        // Interactions in R (upper triangle)
        const float wRR = static_cast<float>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 0, 2, 2, index, 2));
        for (size_t i = 0; i < r_count; i++)
        {
            for (size_t j = i + 1; j < r_count; j++)
            {
                int idx = get_interaction_index(static_cast<int>(r_buffer[i]), static_cast<int>(r_buffer[j]), num_features, 2);
                interactions[idx] += value * wRR;
            }
        }
        // Cross interactions (E × R)
        const float wER = static_cast<float>(weight_cache.get_weight(num_features, E.num_bits(), R.num_bits(), 1, 1, 2, index, 2));
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

    void any_order_bitset_update(const BitSet E, const BitSet R,
        uint64_t *e_buffer, uint64_t *r_buffer,
        uint64_t e_count, uint64_t r_count,
        float value, SparseInteractionMap &interactions, inter_weights::WeightCache &weight_cache, int num_features, IndexType index, int max_order, int verbose)
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
