// Single-reference fill: the interventional recursion for one reference row r and one
// explained point x, one depth-first pass per tree carrying the path's (feature, membership).
//
// At a node with feature f:
//   * f already in E: only x's child continues;
//   * f already in R: only r's child continues;
//   * x and r route the same way: the common child continues, f joins neither set;
//   * they diverge: x's child continues with f in E, r's child with f in R.
#pragma once

#include <cstdint>
#include <vector>

#include "leaf_update.hpp"
#include "utils.cpp"

namespace interventional
{
    enum Membership : uint8_t
    {
        IN_R = 0,
        IN_E = 1,
        FREE = 2
    };

    // The current path and the leaf event built from it, sized to the ensemble's longest path.
    struct PathScratch
    {
        std::vector<int32_t> feats;    // path features by depth
        std::vector<uint8_t> flags;    // their membership by depth
        std::vector<int32_t> ev_feats; // leaf event: the feature ids of E u R, ascending
        std::vector<uint8_t> ev_in_e;  // leaf event: per feature 1 in E, 0 in R
        explicit PathScratch(int max_depth)
            : feats(static_cast<size_t>(max_depth) + 1), flags(static_cast<size_t>(max_depth) + 1),
              ev_feats(static_cast<size_t>(max_depth) + 1), ev_in_e(static_cast<size_t>(max_depth) + 1)
        {
        }
    };

    // Membership of `feature` on the path [0, depth), or FREE.
    inline uint8_t path_membership(const PathScratch &p, int depth, int64_t feature)
    {
        for (int d = 0; d < depth; ++d)
            if (p.feats[static_cast<size_t>(d)] == feature && p.flags[static_cast<size_t>(d)] != FREE)
                return p.flags[static_cast<size_t>(d)];
        return FREE;
    }

    // Builds the leaf event of the path [0, depth) into ev_feats / ev_in_e and returns the
    // counts n = |E u R|, e = |E|, r = |R|.
    inline void gather_leaf_event(PathScratch &p, int depth, int &n, int &e, int &r)
    {
        n = 0;
        e = 0;
        r = 0;
        for (int d = 0; d < depth; ++d)
        {
            const uint8_t flag = p.flags[static_cast<size_t>(d)];
            if (flag == FREE)
                continue;
            const int32_t f = p.feats[static_cast<size_t>(d)];
            int k = n; // insertion into the sorted prefix
            while (k > 0 && p.ev_feats[static_cast<size_t>(k - 1)] > f)
            {
                p.ev_feats[static_cast<size_t>(k)] = p.ev_feats[static_cast<size_t>(k - 1)];
                p.ev_in_e[static_cast<size_t>(k)] = p.ev_in_e[static_cast<size_t>(k - 1)];
                --k;
            }
            p.ev_feats[static_cast<size_t>(k)] = f;
            p.ev_in_e[static_cast<size_t>(k)] = flag;
            ++n;
            if (flag == IN_E)
                ++e;
            else
                ++r;
        }
    }

    struct SingleFrame
    {
        int64_t node;
        int depth;      // the frame's (feature, flag) sits at path position depth - 1
        int32_t feature;
        uint8_t flag;
    };

    // One tree, one reference row r, one explained point x: emits every reachable leaf's event
    // to `update`. `scale` multiplies the leaf value.
    template <typename Update>
    inline void single_reference_fill(Tree &tree, const double *x, const double *r, double scale, PathScratch &p,
                                      std::vector<SingleFrame> &stack, Update &update)
    {
        stack.clear();
        stack.push_back(SingleFrame{0, 0, -1, FREE});
        while (!stack.empty())
        {
            const SingleFrame frame = stack.back();
            stack.pop_back();
            if (frame.depth > 0)
            {
                p.feats[static_cast<size_t>(frame.depth - 1)] = frame.feature;
                p.flags[static_cast<size_t>(frame.depth - 1)] = frame.flag;
            }
            const int64_t node = frame.node;
            if (tree.children_left[node] == tree.children_right[node])
            {
                int n, e, rr;
                gather_leaf_event(p, frame.depth, n, e, rr);
                if (n == 0)
                    continue; // unconstrained leaf: no interaction term
                update(LeafEvent{p.ev_feats.data(), p.ev_in_e.data(), n, e, rr, tree.leaf_predictions[node] * scale});
                continue;
            }
            const int64_t feature = tree.features[node];
            const bool x_left = tree.goes_left(x[feature], node);
            const bool r_left = tree.goes_left(r[feature], node);
            const int64_t x_child = x_left ? tree.children_left[node] : tree.children_right[node];
            const int64_t r_child = r_left ? tree.children_left[node] : tree.children_right[node];
            const uint8_t member = path_membership(p, frame.depth, feature);
            const int32_t f32 = static_cast<int32_t>(feature);
            const int depth = frame.depth + 1;
            if (member == IN_E)
                stack.push_back(SingleFrame{x_child, depth, f32, FREE});
            else if (member == IN_R)
                stack.push_back(SingleFrame{r_child, depth, f32, FREE});
            else if (x_left == r_left)
                stack.push_back(SingleFrame{x_child, depth, f32, FREE});
            else
            {
                stack.push_back(SingleFrame{x_child, depth, f32, IN_E});
                stack.push_back(SingleFrame{r_child, depth, f32, IN_R});
            }
        }
    }
} // namespace interventional
