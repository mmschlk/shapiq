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
};

struct ParsedForest
{
	std::vector<ParsedTreeArrays> trees;
	int64_t num_class = 1;
	double base_score = 0.0;
};

ParsedForest parse_xgboost_ubjson_to_forest(
	const uint8_t *data,
	size_t size,
	int class_label,
	double margin_base_score);

ParsedForest parse_lightgbm_text_to_forest(
	const char *data,
	size_t size,
	int class_label);

ParsedForest parse_catboost_json_to_forest(
	const char *data,
	size_t size,
	int class_label);
