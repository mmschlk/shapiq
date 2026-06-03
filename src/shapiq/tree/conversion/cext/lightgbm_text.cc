#include "converter.hpp"

#include <cctype>

class StringStream
{
private:
    const char *data;
    size_t pos;
    size_t size;

public:
    StringStream(const char *str, size_t len) : data(str), pos(0), size(len) {}

    void skipWhitespace()
    {
        // Skip any whitespace characters starting from the current position. https://en.cppreference.com/w/cpp/string/byte/isspace.html
        // Also skips \t and \n, which can appear in the String
        while (pos < size && std::isspace(static_cast<unsigned char>(data[pos])))
        {
            pos++;
        }
    }
    void skipLine()
    {
        // Skip characters until the end of the current line (i.e., until a newline character or the end of the string).
        while (pos < size && data[pos] != '\n')
        {
            pos++;
        }
        if (pos < size && data[pos] == '\n')
        {
            pos++; // Skip the newline character as well
        }
    }

    int remapNodeId(int n_internals, int id)
    {
        if (id >= 0)
            return id;               // Leave internal node IDs unchanged
        return (n_internals)-id - 1; // Remap leaf node IDs to be after internal nodes
    }

    void findKey(const char *key)
    {
        // Find the given key in the string, starting from the current position.
        // Advances pos to the character immediately following the key if found.
        size_t key_len = std::strlen(key);
        while (pos < size)
        {
            skipWhitespace();
            if (pos + key_len <= size && std::memcmp(data + pos, key, key_len) == 0)
            {
                pos += key_len;
                return;
            }
            pos++;
        }
        throw std::runtime_error(std::string("Key not found: ") + key);
    }

    bool tryFindKey(const char *key)
    {
        // Find the given key in the string, starting from the current position.
        // Advances pos to the character immediately following the key if found.
        size_t key_len = std::strlen(key);
        while (pos < size)
        {
            skipWhitespace();
            if (pos + key_len <= size && std::memcmp(data + pos, key, key_len) == 0)
            {
                pos += key_len;
                return true;
            }
            pos++;
        }
        return false;
    }

    void parseIntLine(std::vector<int64_t> &output)
    {
        // Parse a line of whitespace-separated integers starting from the current position and append them to the output vector.
        while (pos < size && data[pos] != '\n')
        {
            skipWhitespace();
            if (pos >= size || data[pos] == '\n')
                break;
            char *endptr = nullptr;
            int64_t value = std::strtoll(data + pos, &endptr, 10);
            if (endptr == data + pos)
            {
                throw std::runtime_error("Expected integer at position: " + std::to_string(pos));
            }
            output.push_back(value);
            pos = endptr - data; // Advance pos to the character after the parsed integer. The difference computes the new position relative to the start of the string.
        }
    }

    void parseDoubleLine(std::vector<double> &output)
    {
        // Parse a line of whitespace-separated floating-point numbers starting from the current position and append them to the output vector.
        while (pos < size && data[pos] != '\n')
        {
            skipWhitespace();
            if (pos >= size || data[pos] == '\n')
                break;
            char *endptr = nullptr;
            double value = strtod_c(data + pos, &endptr);
            if (endptr == data + pos)
            {
                throw std::runtime_error("Expected floating-point number at position: " + std::to_string(pos));
            }
            output.push_back(value);
            pos = endptr - data; // Advance pos to the character after the parsed number. The difference computes the new position relative to the start of the string.
        }
    }

    void parseIntSingle(int &value)
    {
        skipWhitespace();
        char *endptr = nullptr;
        value = std::strtol(data + pos, &endptr, 10);
        pos = endptr - data; // Advance pos to the character after the parsed integer. The difference computes the new position relative to the start of the string.
    }

    // Read num_tree_per_iteration from the header (equals num_class for multiclass).
    // Uses a copy so the main stream position is unchanged.
    int readNumTreePerIteration()
    {
        StringStream s = *this;
        size_t key_len = std::strlen("num_tree_per_iteration=");
        while (s.pos < s.size)
        {
            s.skipWhitespace();
            if (s.pos + key_len <= s.size && std::memcmp(s.data + s.pos, "num_tree_per_iteration=", key_len) == 0)
            {
                s.pos += key_len;
                int val = 1;
                s.parseIntSingle(val);
                return val;
            }
            s.pos++;
        }
        return 1;
    }

    ParsedForest extractTreeStructure(int class_label = -1)
    {
        int num_class = readNumTreePerIteration();
        bool filtering = (class_label >= 0) && (num_class > 1);

        ParsedForest forest;
        forest.num_class = static_cast<int64_t>(num_class);
        int tree_id = 0, num_cat = 0, num_leaves = 0, num_nodes = 0, num_internal = 0;

        while (true)
        {
            // Find next tree; pos advances past "Tree=".
            // For excluded trees we continue immediately — tryFindKey will
            // scan forward from inside this tree's text to the next "Tree=".
            if (!tryFindKey("Tree="))
                break;
            parseIntSingle(tree_id);

            bool include = !filtering || (tree_id % num_class == class_label);
            if (!include)
                continue;

            std::vector<int64_t> node_ids, feature_ids, left_children, right_children, default_children, internal_count, decision_types;
            std::vector<double> thresholds, values, leaf_count, internal_values;
            // Get num_leaves
            findKey("num_leaves=");
            parseIntSingle(num_leaves);
            num_internal = num_leaves - 1;
            num_nodes = num_internal + num_leaves; //
            // Get num categorical features (0 for non-categorical models). NOTICE: Currently not support for categorical features.
            findKey("num_cat=");
            parseIntSingle(num_cat);
            // Extract Split feature IDs
            findKey("split_feature=");
            parseIntLine(feature_ids);
            // Skip split_gain line
            skipLine();
            // Extract thresholds
            findKey("threshold=");
            parseDoubleLine(thresholds);
            // Extract decision_type (bitset-encoded split metadata)
            findKey("decision_type=");
            parseIntLine(decision_types);
            // Extract left_children
            findKey("left_child=");
            parseIntLine(left_children);
            // Extract right_children
            findKey("right_child=");
            parseIntLine(right_children);
            // Extract Leaf_values
            findKey("leaf_value=");
            parseDoubleLine(values);
            // Skip leaf weight
            skipLine();
            // Extract node_sample_weights
            findKey("leaf_count=");
            parseDoubleLine(leaf_count);
            // Extract internal node values
            findKey("internal_value=");
            parseDoubleLine(internal_values);
            // Skip Internal weight
            skipLine();
            // Get Internal count
            findKey("internal_count=");
            parseIntLine(internal_count);
            // Skip is_linear
            skipLine();
            // Skip shrinkage value
            skipLine();

            ParsedTreeArrays tree;
            tree.node_ids.resize(num_nodes, 0);
            tree.feature_ids.resize(num_nodes, -1);
            tree.thresholds.resize(num_nodes, 0.0);
            tree.values.resize(num_nodes, 0.0);
            tree.left_children.resize(num_nodes, -1);
            tree.right_children.resize(num_nodes, -1);
            tree.default_children.resize(num_nodes, -1);
            tree.node_sample_weights.resize(num_nodes, 0.0);

            if (feature_ids.size() < static_cast<size_t>(num_internal) ||
                thresholds.size() < static_cast<size_t>(num_internal) ||
                left_children.size() < static_cast<size_t>(num_internal) ||
                right_children.size() < static_cast<size_t>(num_internal) ||
                internal_count.size() < static_cast<size_t>(num_internal) ||
                decision_types.size() < static_cast<size_t>(num_internal) ||
                internal_values.size() < static_cast<size_t>(num_internal) ||
                values.size() < static_cast<size_t>(num_leaves) ||
                leaf_count.size() < static_cast<size_t>(num_leaves))
            {
                throw std::runtime_error("LightGBM tree field length mismatch");
            }

            // Process the internal nodes. We can just copy the split feature IDs, thresholds, left/right children, and sample weights. The default child is always the left child in LightGBM.
            for (int i = 0; i < num_internal; i++)
            {
                tree.node_ids[i] = i;
                tree.feature_ids[i] = feature_ids[i];
                tree.thresholds[i] = thresholds[i];
                tree.values[i] = internal_values[i];
                tree.left_children[i] = remapNodeId(num_internal, left_children[i]);
                tree.right_children[i] = remapNodeId(num_internal, right_children[i]);
                bool default_left = (decision_types[i] & 2) != 0;
                tree.default_children[i] = default_left
                                               ? remapNodeId(num_internal, left_children[i])
                                               : remapNodeId(num_internal, right_children[i]);
                tree.node_sample_weights[i] = internal_count[i];
            }
            // Process the leaf nodes.
            for (int i = num_internal; i < num_nodes; i++)
            {
                tree.node_ids[i] = i;
                tree.values[i] = values[i - num_internal];
                tree.node_sample_weights[i] = leaf_count[i - num_internal];
            }

            if (include)
                forest.trees.push_back(std::move(tree));
        }
        return forest;
    }
};

ParsedForest parse_lightgbm_text_to_forest(
	const char *data,
	size_t size,
	int class_label)
{
	StringStream stream(data, size);
	return stream.extractTreeStructure(class_label);
}
