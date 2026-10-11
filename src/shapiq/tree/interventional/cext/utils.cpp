#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef UTILS_H
#define UTILS_H

enum class IndexType
{
    SII,
    BII,
    CHII,
    FBII,
    FSII,
    STII,
    CUSTOM
};

enum DecisionType
{
    LESS_EQUAL = 0,
    LESS_THAN = 1
};

class Tree
{
public:
    Tree(double *leaf_predictions,
         double *thresholds,
         int64_t *features,
         int64_t *children_left,
         int64_t *children_right,
         bool *children_missing,
         std::string decision_type,
         // categorical splits: the node's sorted category set is cat_values[cat_start[node], +cat_size[node]);
         // cat_size[node] == 0 marks a numeric node, nullptr means numeric only
         const int64_t *cat_values = nullptr,
         const int64_t *cat_start = nullptr,
         const int64_t *cat_size = nullptr)
    {
        this->leaf_predictions = leaf_predictions;
        this->thresholds = thresholds;
        this->features = features;
        this->children_left = children_left;
        this->children_right = children_right;
        this->children_missing = children_missing;
        this->cat_values = cat_values;
        this->cat_start = cat_start;
        this->cat_size = cat_size;

        this->decision_type = decision_type == "<=" ? DecisionType::LESS_EQUAL : DecisionType::LESS_THAN;
    }
    DecisionType decision_type;
    double *leaf_predictions;
    double *thresholds;
    int64_t *features;
    int64_t *children_left;
    int64_t *children_right;
    bool *children_missing;
    const int64_t *cat_values;
    const int64_t *cat_start;
    const int64_t *cat_size;

    bool is_leaf(int64_t node_id) const
    {
        return this->children_left[node_id] == -1 && this->children_right[node_id] == -1;
    }

    bool goes_left(double feature_value, int64_t node_id)
    {
        if (std::isnan(feature_value))
        {
            return this->children_missing[node_id];
        }

        if (this->cat_size != nullptr && this->cat_size[node_id] > 0)
        {
            // categorical split: a category in the node's set goes left
            const int64_t category = static_cast<int64_t>(feature_value);
            const int64_t *begin = this->cat_values + this->cat_start[node_id];
            return std::binary_search(begin, begin + this->cat_size[node_id], category);
        }

        if (this->decision_type == DecisionType::LESS_EQUAL)
        {
            return feature_value <= this->thresholds[node_id];
        }
        else if (this->decision_type == DecisionType::LESS_THAN)
        {
            return feature_value < this->thresholds[node_id];
        }
        else
        {
            throw std::invalid_argument("Unsupported decision type: " + std::to_string(static_cast<int>(this->decision_type)));
        }
        return feature_value <= this->thresholds[node_id];
    }
};



#endif
