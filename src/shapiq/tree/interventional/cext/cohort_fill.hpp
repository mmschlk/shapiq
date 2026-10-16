// Cohort fill: one depth-first pass per tree over all reference rows at once.
//
// A frame holds the cohort of reference rows that reached a node with the same (E, R), as a
// slice [begin, end) of the per-tree row order. At a split the cohort is partitioned into the
// rows that route like the explained point (E and R unchanged) and the rows that diverge
// (x's child with the feature in E, the rows' child with it in R). At a leaf the update runs
// once with the leaf value scaled by the cohort size.
//
// A frame stores only the (feature, membership) it added at its depth; the path array holds
// the entries of the frame being processed and of its ancestors, which the LIFO stack keeps in
// place. `member` mirrors the path per feature.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "leaf_update.hpp"
#include "single_reference_fill.hpp"
#include "utils.cpp"

namespace interventional
{
    struct CohortFrame
    {
        int64_t node;
        int begin; // the cohort: slice [begin, end) of ref_order
        int end;
        int depth;   // the frame's (feature, flag) sits at path position depth - 1
        int32_t feature;
        uint8_t flag;
    };

    // Scratch of one cohort pass, sized to the ensemble's longest path.
    struct CohortScratch
    {
        PathScratch path;
        std::vector<uint8_t> member;   // per feature: IN_E / IN_R / FREE on the current path
        std::vector<int> ref_order;    // reference row ids, partitioned in place
        std::vector<CohortFrame> stack;
        int path_depth = 0;            // valid prefix of the path arrays
        CohortScratch(int max_depth, int n_features, int n_ref)
            : path(max_depth), member(static_cast<size_t>(n_features), FREE), ref_order(static_cast<size_t>(n_ref))
        {
            stack.reserve(256);
        }
    };

    // Truncates the path to `depth` entries (clearing the membership of the dropped features)
    // and appends the frame's delta.
    inline void restore_path(CohortScratch &s, const CohortFrame &frame)
    {
        for (int d = frame.depth - 1; d < s.path_depth; ++d)
            if (d >= 0 && s.path.flags[static_cast<size_t>(d)] != FREE)
                s.member[static_cast<size_t>(s.path.feats[static_cast<size_t>(d)])] = FREE;
        if (frame.depth > 0)
        {
            s.path.feats[static_cast<size_t>(frame.depth - 1)] = frame.feature;
            s.path.flags[static_cast<size_t>(frame.depth - 1)] = frame.flag;
            if (frame.flag != FREE)
                s.member[static_cast<size_t>(frame.feature)] = frame.flag;
        }
        s.path_depth = frame.depth;
    }

    // One tree over all reference rows: emits one leaf event per (leaf, cohort).
    // reference_data: n_ref x n_features; x: the explained point; scale: 1 / n_ref, applied
    // with the cohort size to the leaf value.
    template <typename Update>
    inline void cohort_fill(Tree &tree, const double *reference_data, const double *x, int n_ref, int n_features,
                            double scale, CohortScratch &s, Update &update)
    {
        for (int i = 0; i < n_ref; ++i)
            s.ref_order[static_cast<size_t>(i)] = i;
        // `member` is all FREE here: the previous tree cleared its last path on the way out
        s.path_depth = 0;
        s.stack.clear();
        s.stack.push_back(CohortFrame{0, 0, n_ref, 0, -1, FREE});
        while (!s.stack.empty())
        {
            const CohortFrame frame = s.stack.back();
            s.stack.pop_back();
            restore_path(s, frame);
            const int64_t node = frame.node;
            if (tree.children_left[node] == tree.children_right[node])
            {
                int n, e, r;
                gather_leaf_event(s.path, frame.depth, n, e, r);
                if (n == 0)
                    continue;
                const double value = tree.leaf_predictions[node] * static_cast<double>(frame.end - frame.begin) * scale;
                update(LeafEvent{s.path.ev_feats.data(), s.path.ev_in_e.data(), n, e, r, value});
                continue;
            }
            const int64_t feature = tree.features[node];
            const bool x_left = tree.goes_left(x[feature], node);
            const int64_t x_child = x_left ? tree.children_left[node] : tree.children_right[node];
            const int64_t other = x_left ? tree.children_right[node] : tree.children_left[node];
            const int32_t f32 = static_cast<int32_t>(feature);
            const int depth = frame.depth + 1;
            const uint8_t member = s.member[static_cast<size_t>(feature)];
            if (member == IN_E)
            {
                // fixed by the explained point: the whole cohort follows x
                s.stack.push_back(CohortFrame{x_child, frame.begin, frame.end, depth, f32, FREE});
                continue;
            }
            // split the cohort: [begin, mid) routes like x, [mid, end) diverges
            const double *ref_col = reference_data + feature;
            int *slice_begin = s.ref_order.data() + frame.begin;
            int *slice_end = s.ref_order.data() + frame.end;
            int *mid_ptr = std::partition(slice_begin, slice_end, [&](int ref_idx)
                                          { return tree.goes_left(ref_col[static_cast<int64_t>(ref_idx) * n_features], node) == x_left; });
            const int mid = frame.begin + static_cast<int>(mid_ptr - slice_begin);
            if (member == IN_R)
            {
                // fixed by the reference rows: each row follows its own route, f joins nothing
                if (mid > frame.begin)
                    s.stack.push_back(CohortFrame{x_child, frame.begin, mid, depth, f32, FREE});
                if (mid < frame.end)
                    s.stack.push_back(CohortFrame{other, mid, frame.end, depth, f32, FREE});
                continue;
            }
            if (mid > frame.begin) // agreeing cohort: E, R unchanged
                s.stack.push_back(CohortFrame{x_child, frame.begin, mid, depth, f32, FREE});
            if (mid < frame.end) // diverging cohort: both branches
            {
                s.stack.push_back(CohortFrame{x_child, mid, frame.end, depth, f32, IN_E});
                s.stack.push_back(CohortFrame{other, mid, frame.end, depth, f32, IN_R});
            }
        }
        // clear the last path's entries so `member` is all FREE for the next tree
        for (int d = 0; d < s.path_depth; ++d)
            if (s.path.flags[static_cast<size_t>(d)] != FREE)
                s.member[static_cast<size_t>(s.path.feats[static_cast<size_t>(d)])] = FREE;
        s.path_depth = 0;
    }
} // namespace interventional
