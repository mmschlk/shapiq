// The leaf update of the interventional kernels: one leaf event adds, for every interaction S
// among its feature ids up to max_order, weight(|E|, |R|, |S n E|, |S n R|) * value at the
// interaction's position in the output array.
//
// Positions come from the lexicographic rank (complete order-2/3 tables), from the hash index,
// or from a search of the table (no index). The order and the addressing modes are template
// parameters; dispatch_leaf_update picks the instantiation once per call.
#pragma once

#include <cstdint>
#include <vector>

#include "../../cext/subset_tables.hpp"

namespace interventional
{
    // The output array a fill writes into: the order-1 block [0, n_features), then per order
    // >= 2 one entry per table entry at order_offsets[order] + position.
    struct Layout
    {
        int n_features = 0;
        int max_order = 0;
        const int64_t *counts = nullptr;        // per order: entries in its table
        const int64_t *order_offsets = nullptr; // per order: start of its block in the output array
        const int32_t *keys = nullptr;          // the tables (ascending feature ids)
        const int64_t *table_starts = nullptr;  // per order: start of its table in `keys`
        bool has_index = false;                 // the hash index exists (the keys fit)
        subset_tables::SubsetIndex index;       // hash index from key to position
        bool dense2 = false;                    // the order-2 table is complete: position = rank
        bool dense3 = false;                    // the order-3 table is complete: position = rank
        const int64_t *rank3_base = nullptr;    // rank3_base[i] = number of triples with smallest feature < i

        int64_t row(subset_tables::Key128 key, int s) const { return index.row(key, s, counts); }
        // the unrolled orders build int64 keys, which fit while n_features^3 < 2^63
        int64_t row(int64_t key, int s) const { return index.row(subset_tables::Key128{0, static_cast<uint64_t>(key)}, s, counts); }
    };

    // Feature counts below this let the unrolled orders 1-3 build their keys in int64.
    constexpr int kMaxUnrolledFeatures = 1 << 21;

    // A leaf reached with the feature sets E (fixed by the explained point) and R (fixed by
    // the reference). Both fills emit these; the leaf update consumes them.
    struct LeafEvent
    {
        const int32_t *feats; // the feature ids of E u R, ascending
        const uint8_t *in_e;  // per feature: 1 in E, 0 in R
        int n;                // |E u R|
        int e;                // |E|
        int r;                // |R|
        double value;         // leaf value times the cohort size, divided by the number of reference rows
    };

    // Lexicographic rank of the sorted pair (i, j) among all pairs of [0, n).
    inline int64_t pair_rank(int64_t i, int64_t j, int64_t n)
    {
        return i * n - i * (i + 1) / 2 + (j - i - 1);
    }

    // base[i] = number of triples of [0, n) whose smallest feature is < i.
    inline std::vector<int64_t> make_rank3_base(int64_t n)
    {
        std::vector<int64_t> base(static_cast<size_t>(n > 0 ? n : 1), 0);
        for (int64_t i = 1; i < n; ++i)
        {
            const int64_t rest = n - i;
            base[static_cast<size_t>(i)] = base[static_cast<size_t>(i - 1)] + rest * (rest - 1) / 2;
        }
        return base;
    }

    template <bool ATOMIC>
    inline void add_to(double *slot, double value)
    {
        if constexpr (ATOMIC)
        {
#pragma omp atomic
            *slot += value;
        }
        else
        {
            *slot += value;
        }
    }

    // ORDER_CLASS: 1, 2 or 3 for the unrolled orders, 4 for max_order >= 4 (recursive
    // enumeration). DENSE2/DENSE3: the order-2/3 table is complete. SEARCH: no hash index
    // exists, positions come from a search of the table. ATOMIC: `out` is shared between threads.
    template <int ORDER_CLASS, bool DENSE2, bool DENSE3, bool SEARCH, bool ATOMIC>
    struct LeafUpdate
    {
        const Layout &layout;
        const double *weights; // build_weight_table: [((e * stride + r) * k1 + sce) * k1 + scr]
        int stride;
        double *out;
        bool missing = false; // an interaction was not in its table
        subset_tables::TableSearch search; // position lookup when no hash index exists
        std::vector<int32_t> chosen;       // the feature ids chosen so far by `enumerate`

        LeafUpdate(const Layout &layout_, const double *weights_, int stride_, double *out_)
            : layout(layout_), weights(weights_), stride(stride_), out(out_),
              search(layout_.keys, layout_.table_starts, layout_.counts, layout_.max_order),
              chosen(static_cast<size_t>(layout_.max_order > 0 ? layout_.max_order : 1), 0)
        {
        }

        void operator()(const LeafEvent &ev)
        {
            const int k1 = layout.max_order + 1;
            const double *w_er = weights + (static_cast<size_t>(ev.e) * stride + ev.r) * k1 * k1;
            if constexpr (ORDER_CLASS >= 4)
            {
                enumerate(ev, 0, 0, 0, subset_tables::Key128{}, w_er, k1);
                return;
            }
            const int64_t F = layout.n_features;
            const int n = ev.n;
            double *out2 = (ORDER_CLASS >= 2) ? out + layout.order_offsets[2] : nullptr;
            double *out3 = (ORDER_CLASS >= 3) ? out + layout.order_offsets[3] : nullptr;
            for (int i = 0; i < n; ++i)
            {
                const int32_t fi = ev.feats[i];
                const int ei = ev.in_e[i];
                const double v1 = ev.value * w_er[ei * k1 + (1 - ei)];
                if (v1 != 0.0)
                    add_to<ATOMIC>(out + fi, v1);
                if constexpr (ORDER_CLASS < 2)
                    continue;
                const int64_t key_i = static_cast<int64_t>(fi) * F;
                const int64_t pair_base = key_i - static_cast<int64_t>(fi) * (fi + 1) / 2 - fi - 1;
                const int64_t m = F - 1 - fi; // features after fi
                for (int j = i + 1; j < n; ++j)
                {
                    const int32_t fj = ev.feats[j];
                    const int eij = ei + ev.in_e[j];
                    const double v2 = ev.value * w_er[eij * k1 + (2 - eij)];
                    if (v2 != 0.0)
                    {
                        if constexpr (DENSE2)
                            add_to<ATOMIC>(out2 + pair_base + fj, v2);
                        else
                        {
                            const int32_t pair[2] = {fi, fj};
                            probe_add(2, key_i + fj, pair, v2);
                        }
                    }
                    if constexpr (ORDER_CLASS < 3)
                        continue;
                    const int64_t key_ij_f = (key_i + fj) * F;
                    const int64_t a = fj - fi - 1;
                    const int64_t triple_base = DENSE3 ? layout.rank3_base[fi] + a * m - a * (a + 1) / 2 - fi - a - 2 : 0;
                    for (int k = j + 1; k < n; ++k)
                    {
                        const int32_t fk = ev.feats[k];
                        const int eijk = eij + ev.in_e[k];
                        const double v3 = ev.value * w_er[eijk * k1 + (3 - eijk)];
                        if (v3 != 0.0)
                        {
                            if constexpr (DENSE3)
                                add_to<ATOMIC>(out3 + triple_base + fk, v3);
                            else
                            {
                                const int32_t triple[3] = {fi, fj, fk};
                                probe_add(3, key_ij_f + fk, triple, v3);
                            }
                        }
                    }
                }
            }
        }

    private:
        // Adds `value` to the order-s interaction with this key and these ascending feature
        // ids: through the hash index when it exists, by a search of the table otherwise.
        template <typename Key>
        inline void probe_add(int s, Key key, const int32_t *features, double value)
        {
            int64_t row;
            if constexpr (SEARCH)
                row = search.position(features, s);
            else
                row = layout.row(key, s);
            if (row >= 0)
                add_to<ATOMIC>(out + layout.order_offsets[s] + row, value);
            else
                missing = true;
        }

        // Orders 4 and up: recursive enumeration with `level` feature ids chosen so far (sce of
        // them in E), their key, and the first two chosen ids f0, f1 for the ranked orders.
        void enumerate(const LeafEvent &ev, int start, int level, int sce, subset_tables::Key128 key, const double *w_er,
                       int k1, int32_t f0 = 0, int32_t f1 = 0)
        {
            for (int i = start; i < ev.n; ++i)
            {
                const int32_t fi = ev.feats[i];
                const int s = level + 1;
                if constexpr (SEARCH)
                    chosen[static_cast<size_t>(level)] = fi;
                const int sce_new = sce + ev.in_e[i];
                const subset_tables::Key128 key_new =
                    subset_tables::key_mul_add(key, static_cast<uint64_t>(layout.n_features), static_cast<uint64_t>(fi));
                const double w = w_er[sce_new * k1 + (s - sce_new)];
                if (w != 0.0)
                {
                    const double v = ev.value * w;
                    if (s == 1)
                        add_to<ATOMIC>(out + fi, v);
                    else if (s == 2 && DENSE2)
                        add_to<ATOMIC>(out + layout.order_offsets[2] + pair_rank(f0, fi, layout.n_features), v);
                    else if (s == 3 && DENSE3)
                    {
                        const int64_t F = layout.n_features, a = f1 - f0 - 1, m = F - 1 - f0;
                        add_to<ATOMIC>(out + layout.order_offsets[3] + layout.rank3_base[f0] + a * m - a * (a + 1) / 2 - f0 - a - 2 + fi, v);
                    }
                    else
                        probe_add(s, key_new, chosen.data(), v);
                }
                if (s < layout.max_order)
                    enumerate(ev, i + 1, s, sce_new, key_new, w_er, k1, s == 1 ? fi : f0, s == 2 ? fi : f1);
            }
        }
    };

    // Runs `fill(update)` with the LeafUpdate instantiation matching the call. Returns 0, 1 when
    // the update flagged a missing interaction, 2 when no instantiation matches (internal error).
    template <typename Fill>
    inline int dispatch_leaf_update(const Layout &layout, const double *weights, int stride, double *out, bool atomic,
                                    Fill &&fill)
    {
        const int order_class = (layout.max_order >= 4 || layout.n_features >= kMaxUnrolledFeatures) ? 4 : layout.max_order;
        int status = 2;
        auto run = [&](auto tag)
        {
            using U = decltype(tag);
            U update(layout, weights, stride, out);
            fill(update);
            status = update.missing ? 1 : 0;
        };
#define SHAPIQ_LEAF_UPDATE_CASE(OC, D2, D3)                                               \
    if (order_class == OC && layout.dense2 == D2 && layout.dense3 == D3)                  \
    {                                                                                     \
        if (layout.has_index && atomic)                                                   \
            run(LeafUpdate<OC, D2, D3, false, true>(layout, weights, stride, out));       \
        else if (layout.has_index)                                                        \
            run(LeafUpdate<OC, D2, D3, false, false>(layout, weights, stride, out));      \
        else if (atomic)                                                                  \
            run(LeafUpdate<OC, D2, D3, true, true>(layout, weights, stride, out));        \
        else                                                                              \
            run(LeafUpdate<OC, D2, D3, true, false>(layout, weights, stride, out));       \
        return status;                                                                    \
    }
        SHAPIQ_LEAF_UPDATE_CASE(1, false, false)
        SHAPIQ_LEAF_UPDATE_CASE(2, false, false)
        SHAPIQ_LEAF_UPDATE_CASE(2, true, false)
        SHAPIQ_LEAF_UPDATE_CASE(3, false, false)
        SHAPIQ_LEAF_UPDATE_CASE(3, true, false)
        SHAPIQ_LEAF_UPDATE_CASE(3, false, true)
        SHAPIQ_LEAF_UPDATE_CASE(3, true, true)
        SHAPIQ_LEAF_UPDATE_CASE(4, false, false)
        SHAPIQ_LEAF_UPDATE_CASE(4, true, false)
        SHAPIQ_LEAF_UPDATE_CASE(4, false, true)
        SHAPIQ_LEAF_UPDATE_CASE(4, true, true)
#undef SHAPIQ_LEAF_UPDATE_CASE
        return status;
    }
} // namespace interventional
