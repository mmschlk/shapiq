#include "converter.hpp"

#include <cctype>

struct CatBoostSplit
{
    int64_t feature_id = -1;
    double border = 0.0;
};

struct CatBoostObliviousTree
{
    std::vector<CatBoostSplit> splits;
    std::vector<double> leaf_values;
    std::vector<double> leaf_weights;
};

struct CatBoostJsonModel
{
    std::unordered_map<int64_t, std::string> nan_treatments;
    std::vector<CatBoostObliviousTree> trees;
    std::vector<double> bias_values{0.0};
    double scaling = 1.0;
};

class JsonStream
{
private:
    const char *data;
    size_t pos;
    size_t size;

public:
    JsonStream(const char *data, size_t size) : data(data), pos(0), size(size) {}

    void skipWhitespace()
    {
        while (pos < size && std::isspace(static_cast<unsigned char>(data[pos])))
            pos++;
    }

    bool consume(char expected)
    {
        skipWhitespace();
        if (pos < size && data[pos] == expected)
        {
            pos++;
            return true;
        }
        return false;
    }

    void expect(char expected)
    {
        if (!consume(expected))
            throw std::runtime_error(std::string("CatBoost JSON parser expected '") + expected + "'.");
    }

    std::string readString()
    {
        skipWhitespace();
        expect('"');
        std::string out;
        while (pos < size)
        {
            char c = data[pos++];
            if (c == '"')
                return out;
            if (c != '\\')
            {
                out.push_back(c);
                continue;
            }
            if (pos >= size)
                throw std::runtime_error("Invalid escape at end of CatBoost JSON string.");
            char escaped = data[pos++];
            switch (escaped)
            {
            case '"':
            case '\\':
            case '/':
                out.push_back(escaped);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u':
                if (pos + 4 > size)
                    throw std::runtime_error("Invalid unicode escape in CatBoost JSON string.");
                out.append("\\u", 2);
                out.append(data + pos, 4);
                pos += 4;
                break;
            default:
                throw std::runtime_error("Invalid escape in CatBoost JSON string.");
            }
        }
        throw std::runtime_error("Unterminated CatBoost JSON string.");
    }

    double readNumber()
    {
        skipWhitespace();
        const char *start = data + pos;
        char *end = nullptr;
        double value = strtod_c(start, &end);
        if (end == start)
            throw std::runtime_error("Expected numeric CatBoost JSON value.");
        pos += static_cast<size_t>(end - start);
        return value;
    }

    int64_t readInt()
    {
        double value = readNumber();
        return static_cast<int64_t>(value);
    }

    void readLiteral(const char *literal)
    {
        skipWhitespace();
        size_t len = std::strlen(literal);
        if (pos + len > size || std::memcmp(data + pos, literal, len) != 0)
            throw std::runtime_error(std::string("Expected CatBoost JSON literal ") + literal + ".");
        pos += len;
    }

    void skipValue()
    {
        skipWhitespace();
        if (pos >= size)
            throw std::runtime_error("Unexpected end of CatBoost JSON while skipping value.");
        char c = data[pos];
        if (c == '"')
        {
            (void)readString();
            return;
        }
        if (c == '{')
        {
            expect('{');
            if (consume('}'))
                return;
            while (true)
            {
                (void)readString();
                expect(':');
                skipValue();
                if (consume('}'))
                    return;
                expect(',');
            }
        }
        if (c == '[')
        {
            expect('[');
            if (consume(']'))
                return;
            while (true)
            {
                skipValue();
                if (consume(']'))
                    return;
                expect(',');
            }
        }
        if (c == 't')
        {
            readLiteral("true");
            return;
        }
        if (c == 'f')
        {
            readLiteral("false");
            return;
        }
        if (c == 'n')
        {
            readLiteral("null");
            return;
        }
        (void)readNumber(); // Cast to void to indicate we ignore the return value, we just want to advance the position.
    }

    std::vector<double> readNumberArray()
    {
        std::vector<double> values;
        expect('[');
        if (consume(']'))
            return values;
        while (true)
        {
            values.push_back(readNumber());
            if (consume(']'))
                return values;
            expect(',');
        }
    }

    void parseScaleAndBias(CatBoostJsonModel &model)
    {
        expect('[');
        model.scaling = readNumber();
        expect(',');
        model.bias_values = readNumberArray();
        if (model.bias_values.empty())
            throw std::runtime_error("CatBoost scale_and_bias has an empty bias list.");
        expect(']');
    }

    void parseFloatFeature(CatBoostJsonModel &model)
    {
        int64_t feature_index = -1;
        std::string treatment = "AsIs";
        expect('{');
        if (consume('}'))
            return;
        while (true)
        {
            std::string key = readString();
            expect(':');
            if (key == "feature_index")
                feature_index = readInt();
            else if (key == "nan_value_treatment")
                treatment = readString();
            else
                skipValue();
            if (consume('}'))
                break;
            expect(',');
        }
        if (feature_index >= 0)
            model.nan_treatments[feature_index] = treatment;
    }

    void parseFloatFeatures(CatBoostJsonModel &model)
    {
        expect('[');
        if (consume(']'))
            return;
        while (true)
        {
            parseFloatFeature(model);
            if (consume(']'))
                return;
            expect(',');
        }
    }

    void parseFeaturesInfo(CatBoostJsonModel &model)
    {
        expect('{');
        if (consume('}'))
            return;
        while (true)
        {
            std::string key = readString();
            expect(':');
            if (key == "float_features")
                parseFloatFeatures(model);
            else
                skipValue();
            if (consume('}'))
                return;
            expect(',');
        }
    }

    CatBoostSplit parseSplit()
    {
        CatBoostSplit split;
        std::string split_type;
        expect('{');
        if (consume('}'))
            throw std::runtime_error("CatBoost split object is empty.");
        while (true)
        {
            std::string key = readString();
            expect(':');
            if (key == "split_type")
                split_type = readString();
            else if (key == "float_feature_index")
                split.feature_id = readInt();
            else if (key == "border")
                split.border = readNumber();
            else
                skipValue();
            if (consume('}'))
                break;
            expect(',');
        }
        if (split_type != "FloatFeature")
            throw std::runtime_error("Only CatBoost JSON models with FloatFeature splits are supported. Got split_type=" + split_type);
        if (split.feature_id < 0)
            throw std::runtime_error("CatBoost FloatFeature split is missing float_feature_index.");
        return split;
    }

    std::vector<CatBoostSplit> parseSplits()
    {
        std::vector<CatBoostSplit> splits;
        expect('[');
        if (consume(']'))
            return splits;
        while (true)
        {
            splits.push_back(parseSplit());
            if (consume(']'))
                return splits;
            expect(',');
        }
    }

    CatBoostObliviousTree parseTree()
    {
        CatBoostObliviousTree tree;
        bool has_leaf_values = false;
        expect('{');
        if (consume('}'))
            throw std::runtime_error("CatBoost oblivious tree object is empty.");
        while (true)
        {
            std::string key = readString();
            expect(':');
            if (key == "splits")
                tree.splits = parseSplits();
            else if (key == "leaf_values")
            {
                tree.leaf_values = readNumberArray();
                has_leaf_values = true;
            }
            else if (key == "leaf_weights")
                tree.leaf_weights = readNumberArray();
            else
                skipValue();
            if (consume('}'))
                break;
            expect(',');
        }
        if (!has_leaf_values)
            throw std::runtime_error("CatBoost oblivious tree is missing leaf_values.");
        return tree;
    }

    void parseObliviousTrees(CatBoostJsonModel &model)
    {
        expect('[');
        if (consume(']'))
            return;
        while (true)
        {
            model.trees.push_back(parseTree());
            if (consume(']'))
                return;
            expect(',');
        }
    }

    CatBoostJsonModel parseModel()
    {
        CatBoostJsonModel model;
        bool has_trees = false;
        expect('{');
        if (consume('}'))
            throw std::runtime_error("CatBoost JSON root object is empty.");
        while (true)
        {
            std::string key = readString();
            expect(':');
            if (key == "features_info")
                parseFeaturesInfo(model);
            else if (key == "scale_and_bias")
                parseScaleAndBias(model);
            else if (key == "oblivious_trees")
            {
                parseObliviousTrees(model);
                has_trees = true;
            }
            else
                skipValue();
            if (consume('}'))
                break;
            expect(',');
        }
        if (!has_trees)
            throw std::runtime_error("Expected CatBoost JSON model with an 'oblivious_trees' entry.");
        return model;
    }
};

static void fill_catboost_tree_node(
    const std::vector<CatBoostSplit> &splits,
    const std::vector<double> &leaf_values,
    const std::vector<double> &leaf_weights,
    const std::unordered_map<int64_t, std::string> &nan_treatments,
    ParsedTreeArrays &tree,
    uint64_t depth,
    uint64_t internal_count,
    uint64_t level,
    uint64_t leaf_index,
    double scaling,
    double bias)
{
    if (level == depth)
    {
        uint64_t node_id = internal_count + leaf_index;
        tree.node_ids[node_id] = static_cast<int64_t>(node_id);
        tree.feature_ids[node_id] = -2;
        tree.thresholds[node_id] = 0.0;
        tree.values[node_id] = scaling * leaf_values[leaf_index] + bias;
        tree.left_children[node_id] = -1;
        tree.right_children[node_id] = -1;
        tree.default_children[node_id] = -1;
        tree.node_sample_weights[node_id] = leaf_weights[leaf_index];
        return;
    }

    uint64_t node_id = (uint64_t{1} << level) - 1 + leaf_index;
    uint64_t left_child = (uint64_t{1} << (level + 1)) - 1 + leaf_index;
    uint64_t right_leaf_index = leaf_index | (uint64_t{1} << level); // CatBoost oblivious trees use the bits of the leaf index to determine the path, so the right child index is obtained by setting the current level bit.
    uint64_t right_child = (level + 1 == depth)
                               ? internal_count + right_leaf_index
                               : (uint64_t{1} << (level + 1)) - 1 + right_leaf_index;
    const CatBoostSplit &split = splits[level];

    fill_catboost_tree_node(splits, leaf_values, leaf_weights, nan_treatments, tree, depth, internal_count, level + 1, leaf_index, scaling, bias);
    fill_catboost_tree_node(splits, leaf_values, leaf_weights, nan_treatments, tree, depth, internal_count, level + 1, right_leaf_index, scaling, bias);

    tree.node_ids[node_id] = static_cast<int64_t>(node_id);
    tree.feature_ids[node_id] = split.feature_id;
    tree.thresholds[node_id] = split.border;
    tree.values[node_id] = 0.0;
    tree.left_children[node_id] = static_cast<int64_t>(left_child);
    tree.right_children[node_id] = static_cast<int64_t>(right_child);
    auto treatment_it = nan_treatments.find(split.feature_id);
    bool missing_goes_right = treatment_it != nan_treatments.end() && treatment_it->second == "AsTrue";
    tree.default_children[node_id] = missing_goes_right ? static_cast<int64_t>(right_child) : static_cast<int64_t>(left_child);
    tree.node_sample_weights[node_id] = tree.node_sample_weights[left_child] + tree.node_sample_weights[right_child];
}

ParsedForest parse_catboost_json_to_forest(const char *json_data, size_t json_size, int class_label)
{
    JsonStream stream(json_data, json_size);
    CatBoostJsonModel model = stream.parseModel();

    ParsedForest forest;
    size_t tree_count = model.trees.size();
    forest.trees.reserve(tree_count);

    for (const CatBoostObliviousTree &cat_tree : model.trees)
    {
        uint64_t depth = static_cast<uint64_t>(cat_tree.splits.size());
        uint64_t leaf_count = uint64_t{1} << depth;
        uint64_t internal_count = leaf_count - 1;
        uint64_t node_count = internal_count + leaf_count;
        if (cat_tree.leaf_values.empty() || cat_tree.leaf_values.size() % leaf_count != 0)
            throw std::runtime_error("CatBoost leaf_values length is incompatible with the number of leaves.");
        int64_t class_count = static_cast<int64_t>(cat_tree.leaf_values.size() / leaf_count);
        int effective_class_label = class_label;
        if (class_count == 1)
            effective_class_label = 0;
        else if (effective_class_label < 0)
            effective_class_label = 1;
        if (effective_class_label < 0 || effective_class_label >= class_count)
            throw std::runtime_error("CatBoost class_label is outside the available class range.");
        if (static_cast<size_t>(effective_class_label) >= model.bias_values.size())
            throw std::runtime_error("CatBoost scale_and_bias does not contain the selected class bias.");
        double bias_per_tree = model.bias_values[static_cast<size_t>(effective_class_label)] / static_cast<double>(tree_count == 0 ? 1 : tree_count);

        std::vector<double> leaf_values(leaf_count);
        for (uint64_t i = 0; i < leaf_count; ++i)
        {
            size_t leaf_value_index = static_cast<size_t>(i) * static_cast<size_t>(class_count) + static_cast<size_t>(effective_class_label);
            leaf_values[i] = cat_tree.leaf_values[leaf_value_index];
        }

        std::vector<double> leaf_weights(leaf_count, 1.0);
        if (!cat_tree.leaf_weights.empty())
        {
            if (cat_tree.leaf_weights.size() != leaf_count)
                throw std::runtime_error("CatBoost leaf_weights length is incompatible with the number of leaves.");
            leaf_weights = cat_tree.leaf_weights;
        }

        ParsedTreeArrays parsed_tree;
        parsed_tree.node_ids.resize(node_count);
        parsed_tree.feature_ids.resize(node_count, -2);
        parsed_tree.thresholds.resize(node_count, 0.0);
        parsed_tree.values.resize(node_count, 0.0);
        parsed_tree.left_children.resize(node_count, -1);
        parsed_tree.right_children.resize(node_count, -1);
        parsed_tree.default_children.resize(node_count, -1);
        parsed_tree.node_sample_weights.resize(node_count, 0.0);

        if (depth == 0)
        {
            parsed_tree.node_ids[0] = 0;
            parsed_tree.values[0] = model.scaling * leaf_values[0] + bias_per_tree;
            parsed_tree.node_sample_weights[0] = leaf_weights[0];
        }
        else
        {
            fill_catboost_tree_node(cat_tree.splits, leaf_values, leaf_weights, model.nan_treatments, parsed_tree, depth, internal_count, 0, 0, model.scaling, bias_per_tree);
        }
        forest.trees.push_back(std::move(parsed_tree));
    }
    return forest;
}
