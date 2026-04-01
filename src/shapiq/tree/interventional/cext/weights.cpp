#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <tuple>
#include <cmath>
#include "utils.cpp"
#include <string>


namespace inter_weights
{
    /**
     * For the weight computation we deviate from Zern's notation.
     * In our algorithm we keep track of the necessary features:
     * - E: The set of features that are necessary to reach the current node based on the explain point.
     * - R: The set of features that are necessary to reach the current node based on the reference point.
     * We denote the size of E with e and the size of R with r.
     * In Zern's paper, A=E and B=N\R, where N is the set of all features. Thus, we have b = num_features - r.
     */
    constexpr uint64_t binom(signed n, signed k)
    {
        // source: https://github.com/BiagioFesta/algorithms/blob/master/src/DynamicProgramming/BinomialCoefficient.cpp
        if (k < 0)
            return 0;
        if (k > n)
            return 0;
        if (k > n - k)
            k = n - k;
        uint64_t result = 1;
        for (unsigned i = 1; i <= k; ++i)
        {
            result = (result * (n - k + i)) / i;
        }
        return result;
    }
    inline double factorial(int64_t n)
    {
        return (n <= 1) ? 1.0 : static_cast<double>(n) * factorial(n - 1);
    }

    inline double shapley_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        int w1 = e - s_cap_e;
        int w2 = r - s_cap_r;
        return 1.0f / (((w1) + (w2) + 1) * binom(w1 + w2, w1));
    }

    inline double banzhaf_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        return std::ldexp(1.0f, -(e + r - s)); // Equivalent to 1.0f / (2^(a+b))
    }

    inline double chaining_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        // Beta(x,y) = (tgamma(x)*tgamma(y))/tgamma(x+y)
        int w1 = s_cap_r + e;
        int w2 = r - s_cap_r + 1;
        return s * (std::tgamma(w1) * std::tgamma(w2) / std::tgamma(w1 + w2));
    }

    inline double fsii_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        // Compute the lambda
        double w = 0.0;
        for (int i = 0; i <= r - s_cap_r; i++)
        {
            // We can ignore terms where max_order + 1 - u0 - e  < 0 as they contribute 0
            uint64_t denom = binom(e + s_cap_r + i + max_order - 1, max_order + s);
            if (denom == 0) continue;  // numerator is also 0 here; skip to avoid 0/0 = NaN
            w += std::pow(-1.0, s_cap_r + i + max_order - s) * (static_cast<double>(s) / (static_cast<double>(max_order) + static_cast<double>(s)))  * binom(max_order, s) * binom(r - s_cap_r, i) * (static_cast<double>(binom(e + i + s_cap_r - 1, max_order)) / static_cast<double>(denom));
        }
        w += (s_cap_e == e) ? (std::pow(-1.0, s_cap_r)) : 0.0;
        return w;
    }

    inline double fbii_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        // Comput the lambda
        double w = 0.0;
        for (int i = 0; i <= r - s_cap_r; i++)
        {
            // We can ignore terms where max_order + 1 - u0 - e  < 0 as they contribute 0

            w += std::pow(-1.0, s_cap_r + i + max_order - s) * std::ldexp(1.0, -(e + i + s_cap_r - s)) * binom(e + i + s_cap_r - s - 1, max_order - s) * binom(r - s_cap_r, i);
        }
        w += (s_cap_e == e) ? (std::pow(-1.0, s_cap_r)) : 0.0;
        return w;
    }
    inline double discrete_derivative_weight(int64_t coalition_size, int64_t interaction_size, int64_t num_players, IndexType index)
    {
        if (index == IndexType::SII)
        {
            return 1.0f /( (num_players + interaction_size - 1) * binom(num_players-interaction_size, coalition_size));
        }
        if (index == IndexType::BII)
        {
            return std::ldexp(1.0f, -(num_players - interaction_size));
        }
        if (index == IndexType::CHII)
        {
            return static_cast<double>(interaction_size) / (
                static_cast<double>(interaction_size + coalition_size)
                * static_cast<double>(binom(num_players, coalition_size + interaction_size))
            );
        }
        if (index == IndexType::FSII)
        {
            return (
                factorial(2 * interaction_size - 1)
                / std::pow(factorial(interaction_size - 1), 2.0)
                * (
                    factorial(interaction_size + coalition_size - 1)
                    * factorial(num_players - coalition_size - 1)
                    / factorial(num_players + interaction_size - 1)
                )
            );
        }
        if (index == IndexType::STII)
        {
            const double denom = static_cast<double>(binom(num_players - 1, coalition_size));
            if (denom == 0.0)
            {
                return 0.0;
            }
            return (static_cast<double>(interaction_size) / static_cast<double>(num_players)) * (1.0 / denom);
        }
        throw std::invalid_argument("Unsupported index type in discrete_derivative_weight: " + std::to_string(static_cast<int>(index)));
    }
    inline double moebius_weight(int64_t coalition_size, int64_t interaction_size, IndexType index)
    {
        return discrete_derivative_weight(coalition_size - interaction_size, interaction_size, coalition_size, index);
    }
    inline double general_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order, IndexType index)
    {
        double w = 0.0;
        for (int k = 0; k <= r - s_cap_r; k++)
        {
            w += std::pow(-1,k)*binom(r-s_cap_r, k) * moebius_weight(k+s_cap_r + e, s, index);
        }
        return w;
    }
    inline double weight_func(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, IndexType index, int64_t max_order = 1)
    {
        int sign = (s_cap_r % 2 == 0) ? 1 : -1;
        if (index == IndexType::SII)
        {
            return sign * shapley_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        }
        else if (index == IndexType::BII)
        {
            return sign * banzhaf_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        }
        else if (index == IndexType::CHII)
        {
            return sign * chaining_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        }
        else if (index == IndexType::FBII)
        {
            return fbii_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        }
        else
        {
            return sign * general_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order, index);
            //throw std::invalid_argument("Unsupported index type: " + std::to_string(static_cast<int>(index)));
        }
    }

    inline int64_t custom_weight_index(int64_t e, int64_t r, int64_t s_cap_r, int64_t s,
                                        int64_t N, int64_t K)
    {
        // N = n_features + 1, K = max_order + 1
        return e * (N * K * K * K) + r * (K * K) + s_cap_r * K + s;
    }

    // Hash function for tuple-based cache key
    struct CacheKeyHash
    {
        std::size_t operator()(const std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int64_t> &t) const
        {
            // Get all arguments from the tuple and compute individual hashes. Then combine them using XOR and bit rotation to get a single hash value for the entire tuple.
            std::size_t h1 = std::hash<int64_t>{}(std::get<0>(t));
            std::size_t h2 = std::hash<int64_t>{}(std::get<1>(t));
            std::size_t h3 = std::hash<int64_t>{}(std::get<2>(t));
            std::size_t h4 = std::hash<int64_t>{}(std::get<3>(t));
            std::size_t h5 = std::hash<int64_t>{}(std::get<4>(t));
            std::size_t h6 = std::hash<int64_t>{}(std::get<5>(t));
            std::size_t h7 = std::hash<int>{}(std::get<6>(t));
            std::size_t h8 = std::hash<int64_t>{}(std::get<7>(t));

            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6) ^ (h8 << 7);
        }
    };

    class WeightCache
    {
    public:
        using CacheKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int64_t>;
        // Define a HashMap using as key Cachekey (tuple of the arguments of the weight function) and as value the computed weight.
        // The CacheKeyHash struct is used to compute hash values for the tuple keys.
        std::unordered_map<CacheKey, double, CacheKeyHash> cache;

        // Optional custom weight table (nullptr when not in use)
        const double* custom_table;
        int64_t custom_N;  // n_features + 1
        int64_t custom_K;  // max_order + 1

        // Existing constructor (no custom table)
        WeightCache(uint64_t max_number)
            : max_number(max_number), custom_table(nullptr), custom_N(0), custom_K(0)
        {
        }

        // New constructor for custom table
        WeightCache(uint64_t max_number, const double* table, int64_t N, int64_t K)
            : max_number(max_number), custom_table(table), custom_N(N), custom_K(K)
        {
        }

        uint64_t max_number;

        double get_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, IndexType index, int64_t max_order)
        {
            // Early return for custom index: look up directly in the precomputed table
            if (index == IndexType::CUSTOM)
            {
                int64_t idx = custom_weight_index(e, r, s_cap_r, s, custom_N, custom_K);
                return custom_table[idx];
            }
            // Construct the key
            CacheKey key = std::make_tuple(num_features, e, r, s_cap_e, s_cap_r, s, static_cast<int>(index), max_order);
            // Find inherently calls the hash function (CacheKeyHash) to compute the hash value for the key and then looks up the corresponding value in the cache.
            // If the key is found, it returns the cached weight.
            // If the key is not found, it computes the weight using the weight_func, stores it in the cache, and then returns it.
            auto it = cache.find(key);
            if (it != cache.end())
            {
                return it->second;
            }
            else
            {
                double weight = inter_weights::weight_func(num_features, e, r, s_cap_e, s_cap_r, s, index, max_order);
                cache[key] = weight;
                return weight;
            }
        }
    };
}
#endif
