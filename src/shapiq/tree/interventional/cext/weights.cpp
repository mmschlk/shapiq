#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include "utils.cpp"
#include <string>

namespace inter_weights
{
    // E: the features fixed by the explained point on the way to a leaf, R: those fixed by the
    // reference point; e = |E|, r = |R|. In Zern's notation A = E and B = N \ R.
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
        for (signed i = 1; i <= k; ++i)
        {
            result = (result * (n - k + i)) / i;
        }
        return result;
    }
    inline double factorial(int64_t n)
    {
        return (n <= 1) ? 1.0 : static_cast<double>(n) * factorial(n - 1);
    }

    // Every per-index weight function returns the signed lambda, the (-1)^s_cap_r factor
    // included; weight_func is the only dispatch point.
    inline double signed_unit(int64_t s_cap_r)
    {
        return (s_cap_r % 2 == 0) ? 1.0 : -1.0;
    }

    inline double shapley_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        int w1 = e - s_cap_e;
        int w2 = r - s_cap_r;
        return signed_unit(s_cap_r) / (((w1) + (w2) + 1) * binom(w1 + w2, w1));
    }

    inline double banzhaf_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        return signed_unit(s_cap_r) * std::ldexp(1.0f, -(e + r - s)); // Equivalent to 1.0f / (2^(a+b))
    }

    inline double chaining_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        // Beta(x,y) = (tgamma(x)*tgamma(y))/tgamma(x+y)
        int w1 = s_cap_r + e;
        int w2 = r - s_cap_r + 1;
        return signed_unit(s_cap_r) * s * (std::tgamma(w1) * std::tgamma(w2) / std::tgamma(w1 + w2));
    }

    inline double fsii_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        double w = 0.0;
        for (int i = 0; i <= r - s_cap_r; i++)
        {
            uint64_t denom = binom(e + s_cap_r + i + max_order - 1, max_order + s);
            if (denom == 0)
                continue; // the numerator is 0 as well: skip the 0/0 term
            w += std::pow(-1.0, s_cap_r + i + max_order - s) * (static_cast<double>(s) / (static_cast<double>(max_order) + static_cast<double>(s))) * binom(max_order, s) * binom(r - s_cap_r, i) * (static_cast<double>(binom(e + i + s_cap_r - 1, max_order)) / static_cast<double>(denom));
        }
        w += (s_cap_e == e) ? (std::pow(-1.0, s_cap_r)) : 0.0;
        return w;
    }

    inline double fbii_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        double w = 0.0;
        for (int i = 0; i <= r - s_cap_r; i++)
        {
            w += std::pow(-1.0, s_cap_r + i + max_order - s) * std::ldexp(1.0, -(e + i + s_cap_r - s)) * binom(e + i + s_cap_r - s - 1, max_order - s) * binom(r - s_cap_r, i);
        }
        w += (s_cap_e == e) ? (std::pow(-1.0, s_cap_r)) : 0.0;
        return w;
    }

    inline double stii_weight(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, int64_t max_order)
    {
        // Interactions of size s < max_order are the discrete derivative at the empty
        // coalition: the leaf contributes iff E is contained in S, with sign (-1)^|S n R|.
        double w = 0.0;
        if (s < max_order)
        {
            w = (s_cap_e == e) ? std::pow(-1.0, s_cap_r) : 0.0;
        }
        else if (s == max_order)
        {
            int w1 = s + r - s_cap_r;
            int w2 = e - s_cap_e + 1;
            w = std::pow(-1.0, s_cap_r) * s * (std::tgamma(w1) * std::tgamma(w2) / std::tgamma(w1 + w2));
        }
        return w;
    }

    inline double discrete_derivative_weight(int64_t coalition_size, int64_t interaction_size, int64_t num_players, IndexType index)
    {
        if (index == IndexType::SII)
        {
            return 1.0f / ((num_players + interaction_size - 1) * binom(num_players - interaction_size, coalition_size));
        }
        if (index == IndexType::BII)
        {
            return std::ldexp(1.0f, -(num_players - interaction_size));
        }
        if (index == IndexType::CHII)
        {
            return static_cast<double>(interaction_size) / (static_cast<double>(interaction_size + coalition_size) * static_cast<double>(binom(num_players, coalition_size + interaction_size)));
        }
        if (index == IndexType::FSII)
        {
            return (
                factorial(2 * interaction_size - 1) / std::pow(factorial(interaction_size - 1), 2.0) * (factorial(interaction_size + coalition_size - 1) * factorial(num_players - coalition_size - 1) / factorial(num_players + interaction_size - 1)));
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
            w += std::pow(-1, k) * binom(r - s_cap_r, k) * moebius_weight(k + s_cap_r + e, s, index);
        }
        return signed_unit(s_cap_r) * w;
    }

    // The signed leaf weight lambda of an index; the only dispatch point.
    inline double weight_func(int64_t num_features, int64_t e, int64_t r, int64_t s_cap_e, int64_t s_cap_r, int64_t s, IndexType index, int64_t max_order = 1)
    {
        switch (index)
        {
        case IndexType::SII:
            return shapley_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        case IndexType::BII:
            return banzhaf_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        case IndexType::CHII:
            return chaining_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        case IndexType::FBII:
            return fbii_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        case IndexType::FSII:
            return fsii_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        case IndexType::STII:
            return stii_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order);
        default:
            return general_weight(num_features, e, r, s_cap_e, s_cap_r, s, max_order, index);
        }
    }

    inline int64_t custom_weight_index(int64_t e, int64_t r, int64_t s_cap_r, int64_t s,
                                       int64_t N, int64_t K)
    {
        // N = n_features + 1, K = max_order + 1
        return e * (N * K * K * K) + r * (K * K) + s_cap_r * K + s;
    }

    // The weights of one explanation as one array, built per call and indexed
    //   w[((e * stride + r) * k1 + s_cap_e) * k1 + s_cap_r], stride = max_width + 1, k1 = max_order + 1,
    // for every e = |E|, r = |R| with e + r <= max_width (distinct features on the longest path).
    // A custom table (InterventionalTreeSHAPIQ._build_custom_weight_table, custom_N =
    // n_features + 1, custom_K = max_order + 1) is copied into the same shape.
    inline std::vector<double> build_weight_table(IndexType index, int64_t n_features, int max_order, int max_width,
                                                  const double *custom_table, int64_t custom_N, int64_t custom_K)
    {
        const int stride = max_width + 1;
        const int k1 = max_order + 1;
        std::vector<double> w(static_cast<size_t>(stride) * stride * k1 * k1, 0.0);
        for (int e = 0; e <= max_width; ++e)
            for (int r = 0; e + r <= max_width; ++r)
                for (int sce = 0; sce <= std::min(e, max_order); ++sce)
                    for (int scr = 0; scr <= std::min(r, max_order - sce); ++scr)
                    {
                        const int s = sce + scr;
                        if (s == 0)
                            continue;
                        double value;
                        if (index == IndexType::CUSTOM)
                            value = custom_table[custom_weight_index(e, r, scr, s, custom_N, custom_K)];
                        else
                            value = weight_func(n_features, e, r, sce, scr, s, index, max_order);
                        w[((static_cast<size_t>(e) * stride + r) * k1 + sce) * k1 + scr] = value;
                    }
        return w;
    }
}
#endif
