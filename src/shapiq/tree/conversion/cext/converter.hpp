#pragma once

#include <clocale>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Locale-independent strtod: tree model files always use '.' as the decimal
// separator, but std::strtod honours LC_NUMERIC, so a process running under
// e.g. de_DE.UTF-8 would parse "1.5" as 1.0. Force the "C" locale here.
#ifdef _WIN32
static _locale_t C_NUMERIC_LOCALE = _create_locale(LC_NUMERIC, "C");
static inline double strtod_c(const char *s, char **e)
{
	return _strtod_l(s, e, C_NUMERIC_LOCALE);
}
#else
static locale_t C_NUMERIC_LOCALE = newlocale(LC_ALL_MASK, "C", NULL);
static inline double strtod_c(const char *s, char **e)
{
	return strtod_l(s, e, C_NUMERIC_LOCALE);
}
#endif

struct ParsedTreeArrays
{
	std::vector<int64_t> node_ids;
	std::vector<int64_t> feature_ids;
	std::vector<double> thresholds;
	std::vector<double> values;
	std::vector<int64_t> left_children;
	std::vector<int64_t> right_children;
	std::vector<int64_t> default_children;
	std::vector<double> node_sample_weights;
	// categorical splits in CSR layout, mirroring TreeModel: cat_size[node] > 0 marks a
	// categorical node whose left-routed category set is
	// cat_values[cat_start[node] .. cat_start[node] + cat_size[node]).
	// All three stay empty for trees without categorical splits.
	std::vector<int64_t> cat_values;
	std::vector<int64_t> cat_start;
	std::vector<int64_t> cat_size;
};

struct ParsedForest
{
	std::vector<ParsedTreeArrays> trees;
	int64_t num_class = 1;
	double base_score = 0.0;
	// the trees model the class-0 log-odds of a binary classifier (see binary_class_sign)
	bool negated_class_one = false;
};

// Throws unless 0 <= class_label < num_classes. The Python bindings raise
// std::invalid_argument as ValueError; the message matches check_class_label in common.py.
inline void check_class_label(int class_label, int64_t num_classes)
{
	if (class_label < 0 || class_label >= num_classes)
		throw std::invalid_argument(
			"class_label=" + std::to_string(class_label) + " is out of range for a model with " +
			std::to_string(num_classes) + " classes; use 0 to " + std::to_string(num_classes - 1) + ".");
}

// Binary classifiers have a single raw output, the class-1 log-odds; the class-0 log-odds
// are its negation. Returns the sign that selects class_label from that output: -1 for
// class 0, +1 for class 1 or unspecified (-1). Any other label is invalid.
inline double binary_class_sign(int class_label)
{
	if (class_label == -1)
		return 1.0;
	check_class_label(class_label, 2);
	return class_label == 0 ? -1.0 : 1.0;
}

ParsedForest parse_xgboost_ubjson_to_forest(
	const uint8_t *data,
	size_t size,
	int class_label,
	double margin_base_score,
	bool is_classifier);

ParsedForest parse_lightgbm_text_to_forest(
	const char *data,
	size_t size,
	int class_label,
	bool is_classifier);

ParsedForest parse_catboost_json_to_forest(
	const char *data,
	size_t size,
	int class_label,
	bool is_classifier);
