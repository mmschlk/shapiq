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
// multi-word counter over the members of a single coalition (bit b lives in
// word b / 64 at position b % 64), so coalitions of any size k are handled
// without the single-uint64 limit of k <= 63. Global node ids are resolved
// through a hash map keyed by the (sorted) member tuple. Note that the
// enumeration still visits all 2^k subsets, so the practical bound on k is the
// runtime, not the counter width (in practice k <= 22, see the
// max_size_neighbors threshold in graphshapiq.py).

#include <cstdint>
#include <omp.h>
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

    // The subset counter is an array of uint64_t words acting as one arbitrarily
    // wide integer: bit b lives in word b >> 6 (b / 64) at position b & 63
    // (b % 64). Incrementing propagates the carry across word boundaries, so the
    // enumeration over 2^k subsets works for any k, not just k <= 63.

    // Number of words for a counter that must be able to carry bits 0..k. One
    // word more than k bits strictly need, so the termination bit k always
    // exists (also when k % 64 == 0, e.g. k = 64 -> 2 words).
    inline int counter_words(int k)
    {
        return (k >> 6) + 1;
    }

    // counter += 1 with carry propagation across word boundaries.
    inline void counter_increment(uint64_t *words, int n_words)
    {
        for (int w = 0; w < n_words; ++w)
        {
            if (++words[w] != 0)
                break;
        }
    }

    // Test bit b of the counter (b >> 6 selects the word, b & 63 the bit).
    inline bool counter_test_bit(const uint64_t *words, int b)
    {
        return (words[b >> 6] >> (b & 63)) & uint64_t(1);
    }

    // Build the coalition -> prediction-index map from CSR arrays describing
    // `coalition_lookup`.
    //
    // Args:
    //     lookup_members_flat: Concatenated members of the lookup keys.
    //     lookup_offsets: CSR offsets into lookup_members_flat, length n_lookup + 1.
    //     lookup_indices: Prediction row index per lookup key.
    //     n_lookup: Number of lookup keys.
    //
    // Returns:
    //     A hash map from coalition (sorted member vector) to its prediction row index.
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
    // Args:
    //     members_flat: Concatenated members of the coalitions to evaluate.
    //     offsets: CSR offsets into members_flat, length n_coalitions + 1.
    //     n_coalitions: Number of coalitions to evaluate.
    //     lookup_members_flat: Concatenated members of the coalition_lookup keys.
    //     lookup_offsets: CSR offsets into lookup_members_flat, length n_lookup + 1.
    //     lookup_indices: Prediction row index per coalition_lookup key.
    //     n_lookup: Number of coalition_lookup keys.
    //     predictions: coalition_predictions array.
    //     out: Pre-allocated output buffer of length n_coalitions; out[i] receives
    //         the coefficient for coalition i.
    //
    // Returns:
    //     The number of subset lookups that failed (0 on success). A non-zero
    //     value means a subset was missing from the lookup map, which indicates
    //     malformed input.
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

        // Safe to parallelize as-is: each iteration writes only to out[i] (a gather,
        // not a scatter -- see the write-up on GraphSHAPIQ's cpp porting candidates),
        // and only reads from `map`/`predictions`, which are never modified inside the
        // loop. `subset` is declared inside the loop body, so each iteration (and thus
        // each thread) already gets its own instance. `missing` is the only shared
        // mutable state, handled via `reduction(+:missing)`. schedule(dynamic) balances
        // load across threads since coalition sizes k (and thus the 2^k inner loop)
        // vary a lot between coalitions.
#pragma omp parallel for reduction(+ : missing) schedule(dynamic)
        for (int i = 0; i < n_coalitions; ++i)
        {
            const int start = offsets[i];
            const int end = offsets[i + 1];
            const int k = end - start;
            const int *members = members_flat + start;

            double acc = 0.0;
            std::vector<int> subset;
            subset.reserve(static_cast<std::size_t>(k));

            // Counts from 0 through the 2^k local subsets; done once bit k is set.
            std::vector<uint64_t> counter(counter_words(k), 0);
            for (; !counter_test_bit(counter.data(), k);
                 counter_increment(counter.data(), static_cast<int>(counter.size())))
            {
                subset.clear();
                for (int b = 0; b < k; ++b)
                {
                    if (counter_test_bit(counter.data(), b))
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
