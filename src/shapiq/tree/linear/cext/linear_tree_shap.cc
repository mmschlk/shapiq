#include <vector>

// MSVC uses __restrict instead of __restrict__
#ifdef _MSC_VER
  #define __restrict__ __restrict
#endif

struct Tree
{
    double *weights;
    double *leaf_predictions;
    double *thresholds;
    int *parents;
    int *edge_heights;
    int *features;
    int *children_left;
    int *children_right;
    int max_depth;
    int num_nodes;

    Tree(double *weights, double *leaf_predictions, double *thresholds,
         int *parents, int *edge_heights,
         int *features, int *children_left, int *children_right, int max_depth, int num_nodes) : weights(weights), leaf_predictions(leaf_predictions), thresholds(thresholds),
                                                                                                 parents(parents), edge_heights(edge_heights),
                                                                                                 features(features), children_left(children_left),
                                                                                                 children_right(children_right), max_depth(max_depth), num_nodes(num_nodes) {};

    bool is_internal(int pos) const
    {
        return children_left[pos] >= 0;
    }
};

// Stack frame structure for iterative traversal
struct StackFrame
{
    int node;
    int feature;
    int depth;
    int stage; // 0=enter, 1=after_left, 2=after_right, 3=final
};

// Inline psi function (same as recursive version)
inline double psi_iterative(double * __restrict__ e, const double * __restrict__ offset,
                             const double * __restrict__ Base, double q,
                             const double * __restrict__ n, int d)
{
    double res = 0.;
    for (int i = 0; i < d; i++)
    {
        res += e[i] * offset[i] / (Base[i] + q) * n[i];
    }
    return res / d;
}

// Iterative inference function
inline void inference_v2_iterative(
    const Tree &tree,
    const double * __restrict__ Base,
    const double * __restrict__ Offset,
    const double * __restrict__ Norm,
    const double * __restrict__ x,
    bool * __restrict__ activation,
    double * __restrict__ value,
    double * __restrict__ C,
    double * __restrict__ E)
{
    // Cache max_depth as a local to avoid repeated struct-member dereferences
    const int d = tree.max_depth;

    // Allocate stack with sufficient capacity
    const int stack_capacity = d * 5 + 10;
    std::vector<StackFrame> stack(stack_capacity);
    int stack_top = 0;

    // Push initial frame
    stack[stack_top++] = {0, -1, 0, 0};

    // State variables
    double s, q;
    int parent, left, right, offset_degree;
    double *current_e;
    double *child_e;
    double *current_c;
    double *prev_c;
    const double *current_offset;
    const double *current_norm;

    while (stack_top > 0)
    {
        StackFrame current = stack[--stack_top];
        int node = current.node;
        int feature = current.feature;
        int depth = current.depth;
        int stage = current.stage;

        parent = tree.parents[node];
        left = tree.children_left[node];
        right = tree.children_right[node];

        current_e = E + depth * d;
        child_e = E + (depth + 1) * d;
        current_c = C + depth * d;

        if (stage == 0)
        { // Enter node
            s = 0.0;
            if (parent >= 0)
            {
                activation[node] = activation[node] & activation[parent];
                if (activation[parent])
                {
                    s = 1.0 / tree.weights[parent];
                }
            }

            q = 0.0;
            if (feature >= 0)
            {
                if (activation[node])
                {
                    q = 1.0 / tree.weights[node];
                }

                prev_c = C + (depth - 1) * d;
                // Perf: merged two sequential passes into one (branch hoisted outside loop)
                if (parent >= 0) {
                    for (int i = 0; i < d; i++)
                        current_c[i] = prev_c[i] * (Base[i] + q) / (Base[i] + s);
                } else {
                    for (int i = 0; i < d; i++)
                        current_c[i] = prev_c[i] * (Base[i] + q);
                }
            }

            if (left >= 0)
            { // Internal node
                if (x[tree.features[node]] <= tree.thresholds[node])
                {
                    activation[left] = true;
                    activation[right] = false;
                }
                else
                {
                    activation[left] = false;
                    activation[right] = true;
                }

                // Bug fix: use direct indexed writes (vector is pre-sized to stack_capacity)
                // instead of push_back which appended beyond stack_top's tracked region
                stack[stack_top++] = {node, feature, depth, 3};
                stack[stack_top++] = {node, feature, depth, 2};
                stack[stack_top++] = {right, tree.features[node], depth + 1, 0};
                stack[stack_top++] = {node, feature, depth, 1};
                stack[stack_top++] = {left, tree.features[node], depth + 1, 0};
            }
            else
            { // Leaf node
                // Bug fix: initialize current_e before use (parent's stage-1 reads child_e)
                for (int i = 0; i < d; i++) {
                    current_e[i] = current_c[i] * tree.leaf_predictions[node];
                }

                // Process feature contribution immediately for leaf nodes
                if (feature >= 0)
                {
                    if (!(parent >= 0 && !activation[parent]))
                    {
                        q = 0.0;
                        if (activation[node])
                        {
                            q = 1.0 / tree.weights[node];
                        }

                        current_norm = Norm + tree.edge_heights[node] * d;
                        value[feature] += (q - 1.0) * psi_iterative(
                                                          current_e,
                                                          Offset,
                                                          Base,
                                                          q,
                                                          current_norm,
                                                          tree.edge_heights[node]);

                        if (parent >= 0)
                        {
                            s = 0.0;
                            if (activation[parent])
                            {
                                s = 1.0 / tree.weights[parent];
                            }

                            offset_degree = tree.edge_heights[parent] - tree.edge_heights[node];
                            current_norm = Norm + tree.edge_heights[parent] * d;
                            current_offset = Offset + offset_degree * d;
                            value[feature] -= (s - 1.0) * psi_iterative(
                                                              current_e,
                                                              current_offset,
                                                              Base,
                                                              s,
                                                              current_norm,
                                                              tree.edge_heights[parent]);
                        }
                    }
                }
            }
        }
        else if (stage == 1)
        { // After left child
            current_offset = Offset + (tree.edge_heights[node] - tree.edge_heights[left]) * d;
            for (int i = 0; i < d; i++)
            {
                current_e[i] = child_e[i] * current_offset[i];
            }
        }
        else if (stage == 2)
        { // After right child
            current_offset = Offset + (tree.edge_heights[node] - tree.edge_heights[right]) * d;
            for (int i = 0; i < d; i++)
            {
                current_e[i] += child_e[i] * current_offset[i];
            }
        }
        else if (stage == 3)
        { // Final processing for internal nodes
            if (feature >= 0)
            {
                if (parent >= 0 && !activation[parent])
                {
                    continue;
                }

                q = 0.0;
                if (activation[node])
                {
                    q = 1.0 / tree.weights[node];
                }

                current_norm = Norm + tree.edge_heights[node] * d;

                value[feature] += (q - 1.0) * psi_iterative(
                                                  current_e,
                                                  Offset,
                                                  Base,
                                                  q,
                                                  current_norm,
                                                  tree.edge_heights[node]);

                if (parent >= 0)
                {
                    s = 0.0;
                    if (activation[parent])
                    {
                        s = 1.0 / tree.weights[parent];
                    }

                    offset_degree = tree.edge_heights[parent] - tree.edge_heights[node];
                    current_norm = Norm + tree.edge_heights[parent] * d;
                    current_offset = Offset + offset_degree * d;
                    value[feature] -= (s - 1.0) * psi_iterative(
                                                      current_e,
                                                      current_offset,
                                                      Base,
                                                      s,
                                                      current_norm,
                                                      tree.edge_heights[parent]);
                }
            }
        }
    };
}

// Main entry point for iterative version
inline void linear_tree_shap_iterative(
    const Tree &tree,
    const double *Base,
    const double *Offset,
    const double *Norm,
    const double *X,
    int n_row,
    int n_col,
    double *out)
{
    int size = (tree.max_depth + 1) * tree.max_depth;

    // Allocate working buffers once
    double *C = new double[size];
    std::fill_n(C, size, 1.);
    double *E = new double[size];
    bool *activation = new bool[tree.num_nodes];

    // Process all rows
    for (int i = 0; i < n_row; i++)
    {
        const double *x = X + i * n_col;
        double *value = out + i * n_col;
        inference_v2_iterative(tree, Base, Offset, Norm, x, activation, value, C, E);
    }

    delete[] C;
    delete[] E;
    delete[] activation;
}
