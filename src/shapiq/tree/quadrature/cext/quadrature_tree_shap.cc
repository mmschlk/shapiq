#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// MSVC uses __restrict instead of __restrict__
#ifdef _MSC_VER
  #define __restrict__ __restrict
#endif

// Split comparison convention of the converted model; mirrors the linear kernel.
enum QDecisionType
{
    Q_LESS_EQUAL = 0,
    Q_LESS_THAN = 1
};

struct QuadTree
{
    const double *thresholds;
    const int *features;       // split feature per node (-2 at leaves)
    const int *children_left;  // -1 at leaves
    const int *children_right;
    const int *parents;    // tree parent of each node (-1 at the root)
    const int *ancestors;  // closest ancestor node whose incoming edge has the same feature
    const double *c_acc;   // accumulated cold factor (product of cover ratios of the chain)
    const double *values;  // per-node prediction value (leaf values at leaves)
    const int64_t *cat_values;
    const int64_t *cat_start;
    const int64_t *cat_size;
    const unsigned char *children_left_default;
    int max_depth;
    int num_nodes;
    QDecisionType decision_type;

    bool goes_left(double feature_value, int node) const
    {
        if (std::isnan(feature_value))
        {
            return children_left_default[node] != 0;
        }
        if (cat_size[node] > 0)
        {
            const int64_t category = static_cast<int64_t>(feature_value);
            const int64_t *begin = cat_values + cat_start[node];
            return std::binary_search(begin, begin + cat_size[node], category);
        }
        if (decision_type == Q_LESS_THAN)
        {
            return feature_value < thresholds[node];
        }
        return feature_value <= thresholds[node];
    }
};

struct QuadFrame
{
    int node;
    int depth;
    int stage;  // 0 enter, 1 after left, 2 after right, 3 leave
};

// Workspace reused across instances; sized once per kernel invocation.
struct QuadWorkspace
{
    int n_quad;
    int n_feats;
    int min_order;
    int max_order;
    std::vector<double> A;       // (max_depth + 2) x n_quad path products
    std::vector<double> E;       // (max_depth + 2) x n_quad subtree sums
    std::vector<double> live_g;  // n_feats x n_quad current Banzhaf ratios (h - c) / u
    std::vector<double> gamma;   // max_order x n_quad running subset products
    std::vector<double> delta;   // n_quad weighted edge difference w * E * delta
    std::vector<bool> act;       // hot-chain activation per node
    std::vector<int> path_feats;  // sorted distinct features on the current path
    // Sparse interaction support: for each order >= 2 a lexicographically sorted table of the
    // co-occurring subsets (int32 rows of length `order`). Order 1 stays a dense per-feature block.
    const int32_t *subset_keys;
    const int64_t *subset_starts;   // per order: start of its table in subset_keys (in ints)
    const int64_t *subset_counts;   // per order: number of subsets in its table
    const int64_t *order_offsets;   // per order: start of its block in the output
    std::vector<QuadFrame> stack;
    std::vector<int> merged_scratch;
    std::vector<int> candidates;
    std::vector<int> chosen;
    std::vector<int64_t> cursor;  // per order: table position of the last emitted subset

    QuadWorkspace(const QuadTree &tree, int n_quad_, int n_feats_, int min_order_, int max_order_,
                  const int32_t *subset_keys_, const int64_t *subset_starts_,
                  const int64_t *subset_counts_, const int64_t *order_offsets_)
        : n_quad(n_quad_), n_feats(n_feats_), min_order(min_order_), max_order(max_order_),
          subset_keys(subset_keys_), subset_starts(subset_starts_),
          subset_counts(subset_counts_), order_offsets(order_offsets_)
    {
        A.assign(static_cast<size_t>(tree.max_depth + 2) * n_quad, 1.0);
        E.assign(static_cast<size_t>(tree.max_depth + 2) * n_quad, 0.0);
        live_g.assign(static_cast<size_t>(n_feats) * n_quad, 0.0);
        gamma.assign(static_cast<size_t>(std::max(max_order, 1)) * n_quad, 0.0);
        delta.assign(n_quad, 0.0);
        act.assign(tree.num_nodes, false);
        path_feats.reserve(n_feats);
        chosen.resize(static_cast<size_t>(std::max(max_order, 1)));
        cursor.assign(static_cast<size_t>(max_order) + 1, 0);
        stack.reserve(static_cast<size_t>(tree.max_depth) * 5 + 10);
    }

    // Lexicographic compare of a table row against the merged tuple (both length s).
    static int compare_row(const int32_t *row, const int *merged, int s)
    {
        for (int i = 0; i < s; ++i)
        {
            if (row[i] != merged[i])
                return (row[i] < merged[i]) ? -1 : 1;
        }
        return 0;
    }

    // Position (within the order's table) of the sorted tuple formed by merging `feature`
    // into the first `size` chosen path features (chosen[] sorted, feature not among them).
    // Every merged tuple lies on the current path, so it is guaranteed to be in the table.
    // Within one edge extraction the emitted tuples are strictly increasing per order, so the
    // search gallops forward from the per-order cursor instead of bisecting the whole table.
    int64_t merged_position(const int *chosen, int size, int feature)
    {
        merged_scratch.resize(static_cast<size_t>(size) + 1);
        int *merged = merged_scratch.data();
        int insert_at = 0;
        while (insert_at < size && chosen[insert_at] < feature)
            ++insert_at;
        std::memcpy(merged, chosen, static_cast<size_t>(insert_at) * sizeof(int));
        merged[insert_at] = feature;
        std::memcpy(merged + insert_at + 1, chosen + insert_at,
                    static_cast<size_t>(size - insert_at) * sizeof(int));
        const int s = size + 1;
        const int32_t *base = subset_keys + subset_starts[s];
        const int64_t count = subset_counts[s];
        int64_t lo = cursor[s];
        if (lo >= count || compare_row(base + lo * s, merged, s) > 0)
            lo = 0;  // cursor overshot (new extraction); restart from the table head
        // exponential gallop: find a bracket [lo, hi) whose upper row is >= merged
        int64_t bound = 1;
        while (lo + bound < count && compare_row(base + (lo + bound) * s, merged, s) < 0)
            bound <<= 1;
        int64_t hi = std::min(lo + bound + 1, count);
        lo = lo + (bound >> 1);
        // binary search within the bracket
        while (lo < hi)
        {
            const int64_t mid = lo + ((hi - lo) >> 1);
            if (compare_row(base + mid * s, merged, s) < 0)
                lo = mid + 1;
            else
                hi = mid;
        }
        cursor[s] = lo + 1;
        return lo;
    }
};

// Enumerate the subsets of the current path features (excluding `feature`) in sorted order,
// accumulating the gamma products level by level and adding every completed order's
// contribution. `weighted` already carries w * E * delta.
static void quad_enumerate(
    QuadWorkspace &ws,
    const std::vector<int> &candidates,
    size_t start,
    int level,  // number of chosen elements so far
    int *chosen,
    const double *__restrict__ weighted,
    int feature,
    double *__restrict__ out)
{
    const int n_quad = ws.n_quad;
    for (size_t i = start; i < candidates.size(); ++i)
    {
        int j = candidates[i];
        const double *g_j = ws.live_g.data() + static_cast<size_t>(j) * n_quad;
        double *gamma_level = ws.gamma.data() + static_cast<size_t>(level) * n_quad;
        const double *gamma_prev = (level == 0) ? nullptr : ws.gamma.data() + static_cast<size_t>(level - 1) * n_quad;
        for (int m = 0; m < n_quad; ++m)
        {
            gamma_level[m] = (level == 0 ? 1.0 : gamma_prev[m]) * g_j[m];
        }
        chosen[level] = j;
        int order = level + 2;  // subset of size level+1 plus the pivot feature
        if (order >= ws.min_order && order >= 2)
        {
            double contribution = 0.0;
            for (int m = 0; m < n_quad; ++m)
                contribution += weighted[m] * gamma_level[m];
            int64_t position = ws.merged_position(chosen, level + 1, feature);
            out[ws.order_offsets[order] + position] += contribution;
        }
        if (order < ws.max_order)
        {
            quad_enumerate(ws, candidates, i + 1, level + 1, chosen, weighted, feature, out);
        }
    }
}

static void quad_extract(
    const QuadTree &tree,
    QuadWorkspace &ws,
    const double *__restrict__ t,
    const double *__restrict__ w,
    int node,
    int depth,
    double *__restrict__ out)
{
    const int n_quad = ws.n_quad;
    int ancestor = tree.ancestors[node];
    if (ancestor >= 0 && !ws.act[ancestor])
        return;  // both telescoping terms cancel exactly once the hot chain is broken
    int feature = tree.features[tree.parents[node]];
    const double *g_new = ws.live_g.data() + static_cast<size_t>(feature) * n_quad;
    const double *E_row = ws.E.data() + static_cast<size_t>(depth) * n_quad;
    double sum1 = 0.0;
    if (ancestor >= 0)
    {
        double h0 = ws.act[ancestor] ? 1.0 : 0.0;
        double c0 = tree.c_acc[ancestor];
        for (int m = 0; m < n_quad; ++m)
        {
            double u0 = h0 * t[m] + c0 * (1.0 - t[m]);
            ws.delta[m] = w[m] * E_row[m] * (g_new[m] - (h0 - c0) / u0);
            sum1 += ws.delta[m];
        }
    }
    else
    {
        for (int m = 0; m < n_quad; ++m)
        {
            ws.delta[m] = w[m] * E_row[m] * g_new[m];
            sum1 += ws.delta[m];
        }
    }
    if (ws.min_order == 1)
    {
        out[ws.order_offsets[1] + feature] += sum1;
    }
    if (ws.max_order > 1 && !ws.path_feats.empty())
    {
        ws.candidates.clear();
        for (int j : ws.path_feats)
            if (j != feature)
                ws.candidates.push_back(j);
        if (!ws.candidates.empty())
        {
            quad_enumerate(ws, ws.candidates, 0, 0, ws.chosen.data(), ws.delta.data(), feature, out);
        }
    }
}

inline void quadrature_inference(
    const QuadTree &tree,
    QuadWorkspace &ws,
    const double *__restrict__ t,
    const double *__restrict__ w,
    const int *__restrict__ roots,
    int n_trees,
    const double *__restrict__ x,
    double *__restrict__ out)
{
    const int n_quad = ws.n_quad;
    std::fill(ws.act.begin(), ws.act.end(), false);
    ws.path_feats.clear();

    for (int tree_index = 0; tree_index < n_trees; ++tree_index)
    {
    const int root = roots[tree_index];
    std::fill(ws.A.begin(), ws.A.begin() + n_quad, 1.0);
    ws.stack.clear();
    ws.stack.push_back({root, 0, 0});
    while (!ws.stack.empty())
    {
        QuadFrame frame = ws.stack.back();
        ws.stack.pop_back();
        int node = frame.node;
        int depth = frame.depth;
        double *A_row = ws.A.data() + static_cast<size_t>(depth) * n_quad;
        double *E_row = ws.E.data() + static_cast<size_t>(depth) * n_quad;
        const double *child_E = ws.E.data() + static_cast<size_t>(depth + 1) * n_quad;

        if (frame.stage == 0)
        {
            int ancestor = tree.ancestors[node];
            if (node != root)
            {
                if (ancestor >= 0)
                    ws.act[node] = ws.act[node] && ws.act[ancestor];
                int feature = tree.features[tree.parents[node]];
                double h = ws.act[node] ? 1.0 : 0.0;
                double c = tree.c_acc[node];
                const double *A_prev = ws.A.data() + static_cast<size_t>(depth - 1) * n_quad;
                double *g_row = ws.live_g.data() + static_cast<size_t>(feature) * n_quad;
                if (ancestor >= 0)
                {
                    double h0 = ws.act[ancestor] ? 1.0 : 0.0;
                    double c0 = tree.c_acc[ancestor];
                    for (int m = 0; m < n_quad; ++m)
                    {
                        double u_new = h * t[m] + c * (1.0 - t[m]);
                        double u_old = h0 * t[m] + c0 * (1.0 - t[m]);
                        A_row[m] = A_prev[m] * (u_new / u_old);
                        g_row[m] = (h - c) / u_new;
                    }
                }
                else
                {
                    for (int m = 0; m < n_quad; ++m)
                    {
                        double u_new = h * t[m] + c * (1.0 - t[m]);
                        A_row[m] = A_prev[m] * u_new;
                        g_row[m] = (h - c) / u_new;
                    }
                    ws.path_feats.insert(
                        std::lower_bound(ws.path_feats.begin(), ws.path_feats.end(), feature),
                        feature);
                }
            }
            int left = tree.children_left[node];
            int right = tree.children_right[node];
            if (left >= 0)
            {
                bool go_left = tree.goes_left(x[tree.features[node]], node);
                ws.act[left] = go_left;
                ws.act[right] = !go_left;
                ws.stack.push_back({node, depth, 3});
                ws.stack.push_back({node, depth, 2});
                ws.stack.push_back({right, depth + 1, 0});
                ws.stack.push_back({node, depth, 1});
                ws.stack.push_back({left, depth + 1, 0});
            }
            else
            {
                double value = tree.values[node];
                for (int m = 0; m < n_quad; ++m)
                    E_row[m] = A_row[m] * value;
                if (node != root)
                {
                    quad_extract(tree, ws, t, w, node, depth, out);
                    // restore the live state of this edge's feature
                    int feature = tree.features[tree.parents[node]];
                    if (ancestor >= 0)
                    {
                        double h0 = ws.act[ancestor] ? 1.0 : 0.0;
                        double c0 = tree.c_acc[ancestor];
                        double *g_row = ws.live_g.data() + static_cast<size_t>(feature) * n_quad;
                        for (int m = 0; m < n_quad; ++m)
                            g_row[m] = (h0 - c0) / (h0 * t[m] + c0 * (1.0 - t[m]));
                    }
                    else
                    {
                        ws.path_feats.erase(
                            std::lower_bound(ws.path_feats.begin(), ws.path_feats.end(), feature));
                    }
                }
            }
        }
        else if (frame.stage == 1)
        {
            std::memcpy(E_row, child_E, static_cast<size_t>(n_quad) * sizeof(double));
        }
        else if (frame.stage == 2)
        {
            for (int m = 0; m < n_quad; ++m)
                E_row[m] += child_E[m];
        }
        else if (node != root)  // stage 3 on a non-root internal node
        {
            quad_extract(tree, ws, t, w, node, depth, out);
            int ancestor = tree.ancestors[node];
            int feature = tree.features[tree.parents[node]];
            if (ancestor >= 0)
            {
                double h0 = ws.act[ancestor] ? 1.0 : 0.0;
                double c0 = tree.c_acc[ancestor];
                double *g_row = ws.live_g.data() + static_cast<size_t>(feature) * n_quad;
                for (int m = 0; m < n_quad; ++m)
                    g_row[m] = (h0 - c0) / (h0 * t[m] + c0 * (1.0 - t[m]));
            }
            else
            {
                ws.path_feats.erase(
                    std::lower_bound(ws.path_feats.begin(), ws.path_feats.end(), feature));
            }
        }
    }
    }
}

inline void quadrature_tree_shap(
    const QuadTree &tree,
    const double *t,
    const double *w,
    int n_quad,
    const int *roots,
    int n_trees,
    int n_feats,
    int min_order,
    int max_order,
    const int32_t *subset_keys,
    const int64_t *subset_starts,
    const int64_t *subset_counts,
    const int64_t *order_offsets,
    const double *X,
    int n_row,
    int n_col,
    int64_t out_stride,
    double *out)
{
    QuadWorkspace ws(tree, n_quad, n_feats, min_order, max_order,
                     subset_keys, subset_starts, subset_counts, order_offsets);
    for (int i = 0; i < n_row; ++i)
    {
        quadrature_inference(tree, ws, t, w, roots, n_trees,
                             X + static_cast<size_t>(i) * n_col,
                             out + static_cast<size_t>(i) * out_stride);
    }
}
