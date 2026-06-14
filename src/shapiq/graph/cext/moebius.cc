// Core algorithm for the Möbius transform used by GraphSHAP-IQ.
//
// This file contains *no* Python/NumPy code. It is included by cext.cc, which
// handles the Python <-> C++ boundary. The function here mirrors the inner
// double loop of `GraphSHAPIQ.compute_moebius_transform` (Python reference):
//
//     for i, coalition in enumerate(coalitions):
//         for subset in powerset(coalition):
//             sign = (-1) ** (len(coalition) - len(subset))
//             moebius_values[i] += sign * coalition_predictions[coalition_lookup[subset]]
//
// Data is passed in flat (CSR-like) arrays so that arbitrary player indices are
// supported (no 64-player bitmask limit): the subset enumeration uses a *local*
// bitmask over the members of a single coalition (size k, in practice <= 22),
// while global node ids are resolved through a hash map keyed by the (sorted)
// member tuple.

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace moebius
{

    // FNV-1a hash for a vector<int>. Coalition member tuples are stored sorted
    // (the Python side builds them from sorted tuples), so equal coalitions hash
    // identically. std::vector<int> already provides operator==.
    struct VecHash
    {
        std::size_t operator()(const std::vector<int> &v) const
        {
            std::size_t h = 1469598103934665603ULL; // FNV offset basis
            for (int x : v)
            {
                h ^= static_cast<std::size_t>(static_cast<unsigned int>(x));
                h *= 1099511628211ULL; // FNV prime
            }
            return h;
        }
    };

    using LookupMap = std::unordered_map<std::vector<int>, int, VecHash>;

    // Build the coalition -> prediction-index map from CSR arrays describing
    // `coalition_lookup`.
    inline LookupMap build_lookup(
        const int *lookup_members_flat,
        const int *lookup_offsets,
        const int *lookup_indices,
        int n_lookup)
    {
        LookupMap map;
        map.reserve(static_cast<std::size_t>(n_lookup) * 2);
        for (int j = 0; j < n_lookup; ++j)
        {
            const int start = lookup_offsets[j];
            const int end = lookup_offsets[j + 1];
            std::vector<int> key(lookup_members_flat + start, lookup_members_flat + end);
            map.emplace(std::move(key), lookup_indices[j]);
        }
        return map;
    }

    // Compute the Möbius coefficient for every coalition.
    //
    // members_flat / offsets : CSR description of the coalitions to evaluate
    //                          (offsets has length n_coalitions + 1). out[i]
    //                          receives the coefficient for coalition i.
    // lookup_*               : CSR description of coalition_lookup (keys + the
    //                          prediction row index for each key).
    // predictions            : coalition_predictions array.
    // out                    : pre-allocated output buffer of length n_coalitions.
    //
    // Returns the number of subset lookups that failed (0 on success). A
    // non-zero value means a subset was missing from the lookup map, which
    // indicates malformed input.
    inline int compute(
        const int *members_flat,
        const int *offsets,
        int n_coalitions,
        const int *lookup_members_flat,
        const int *lookup_offsets,
        const int *lookup_indices,
        int n_lookup,
        const double *predictions,
        double *out)
    {
        const LookupMap map = build_lookup(
            lookup_members_flat, lookup_offsets, lookup_indices, n_lookup);

        int missing = 0;

        for (int i = 0; i < n_coalitions; ++i)
        {
            const int start = offsets[i];
            const int end = offsets[i + 1];
            const int k = end - start;
            const int *members = members_flat + start;

            double acc = 0.0;
            std::vector<int> subset;
            subset.reserve(static_cast<std::size_t>(k));

            const uint64_t n_subsets = uint64_t(1) << k; // 2^k local subsets
            for (uint64_t local_mask = 0; local_mask < n_subsets; ++local_mask)
            {
                subset.clear();
                for (int b = 0; b < k; ++b)
                {
                    if (local_mask & (uint64_t(1) << b))
                    {
                        subset.push_back(members[b]);
                    }
                }
                const int sub_size = static_cast<int>(subset.size());
                const int sign = ((k - sub_size) & 1) ? -1 : 1;

                auto it = map.find(subset);
                if (it == map.end())
                {
                    ++missing;
                    continue;
                }
                acc += static_cast<double>(sign) * predictions[it->second];
            }

            out[i] = acc;
        }

        return missing;
    }

} // namespace moebius
