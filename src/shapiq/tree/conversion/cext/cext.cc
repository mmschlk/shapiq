#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cstring>

#include "converter.hpp"

static PyObject *parse_xgboost_ubjson(PyObject *self, PyObject *args);
static PyObject *parse_xgboost_ubjson_treemodels(PyObject *self, PyObject *args);
static PyObject *parse_lightgbm_string_treemodels(PyObject *self, PyObject *args);
static PyObject *parse_catboost_json_treemodels(PyObject *self, PyObject *args);
static PyObject *create_edge_tree_arrays(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
	{"parse_xgboost_ubjson", parse_xgboost_ubjson, METH_VARARGS, "Parse XGBoost UBJSON bytes and return tree-structure numpy-array lists."},
	{"parse_xgboost_ubjson_treemodels", parse_xgboost_ubjson_treemodels, METH_VARARGS, "Parse XGBoost UBJSON bytes and return a list of TreeModel objects."},
	{"parse_lightgbm_string_treemodels", parse_lightgbm_string_treemodels, METH_VARARGS, "Parse LightGBM model_to_string() bytes and return a list of TreeModel objects."},
	{"parse_catboost_json_treemodels", parse_catboost_json_treemodels, METH_VARARGS, "Parse CatBoost JSON bytes and return a list of TreeModel objects."},
	{"create_edge_tree_arrays", create_edge_tree_arrays, METH_VARARGS, "Create EdgeTree arrays from a TreeModel representation."},
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"cext",
	"Gradient boosting UBJSON parser C-extension.",
	-1,
	module_methods,
	NULL,
	NULL,
	NULL,
	NULL
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cext(void)
#else
PyMODINIT_FUNC initcext(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
	PyObject *module = PyModule_Create(&moduledef);
	if (!module)
		return NULL;
#else
	PyObject *module = Py_InitModule("cext", module_methods);
	if (!module)
		return;
#endif

	import_array();

#if PY_MAJOR_VERSION >= 3
	return module;
#endif
}

// Cast a double vector to float32 for XGBoost, whose internal DMatrix uses
// float32 thresholds and values. Other backends keep double precision.
static PyArrayObject *numpy_float32_from_double_vector(const std::vector<double> &buffer)
{
	npy_intp dims[1] = {static_cast<npy_intp>(buffer.size())};
	PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(1, dims, NPY_FLOAT));
	if (!arr)
		return NULL;
	float *dst = static_cast<float *>(PyArray_DATA(arr));
	for (size_t i = 0; i < buffer.size(); ++i)
		dst[i] = static_cast<float>(buffer[i]);
	return arr;
}

static PyArrayObject *numpy_float64_from_double_vector(const std::vector<double> &buffer)
{
	npy_intp dims[1] = {static_cast<npy_intp>(buffer.size())};
	PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
	if (!arr)
		return NULL;
	if (!buffer.empty())
		std::memcpy(PyArray_DATA(arr), buffer.data(), buffer.size() * sizeof(double));
	return arr;
}

template <typename T>
static PyArrayObject *numpy_array_from_vector(const std::vector<T> &buffer, int typenum)
{
	npy_intp dims[1] = {static_cast<npy_intp>(buffer.size())};
	PyArrayObject *array = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(1, dims, typenum));
	if (!array)
	{
		return NULL;
	}
	if (!buffer.empty())
	{
		std::memcpy(PyArray_DATA(array), buffer.data(), buffer.size() * sizeof(T));
	}
	return array;
}

template <typename T>
static PyObject *forest_field_to_pylist(const ParsedForest &forest, const std::vector<T> ParsedTreeArrays::*field, int typenum)
{
	PyObject *list = PyList_New(static_cast<Py_ssize_t>(forest.trees.size()));
	if (!list)
	{
		return NULL;
	}
	for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(forest.trees.size()); ++i)
	{
		const std::vector<T> &buffer = forest.trees[static_cast<size_t>(i)].*field;
		PyArrayObject *arr = numpy_array_from_vector<T>(buffer, typenum);
		if (!arr)
		{
			Py_DECREF(list);
			return NULL;
		}
		PyList_SET_ITEM(list, i, reinterpret_cast<PyObject *>(arr));
	}

	return list;
}

static PyObject *forest_double_field_to_float32_pylist(const ParsedForest &forest, const std::vector<double> ParsedTreeArrays::*field)
{
	PyObject *list = PyList_New(static_cast<Py_ssize_t>(forest.trees.size()));
	if (!list)
		return NULL;
	for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(forest.trees.size()); ++i)
	{
		const std::vector<double> &buffer = forest.trees[static_cast<size_t>(i)].*field;
		PyArrayObject *arr = numpy_float32_from_double_vector(buffer);
		if (!arr)
		{
			Py_DECREF(list);
			return NULL;
		}
		PyList_SET_ITEM(list, i, reinterpret_cast<PyObject *>(arr));
	}
	return list;
}

static PyObject *forest_to_treemodel_list(const ParsedForest &forest, const char *decision_type_cstr, bool float32_outputs)
{
	const size_t n_trees = forest.trees.size();
	PyObject *result = PyList_New(static_cast<Py_ssize_t>(n_trees));
	if (!result)
	{
		return NULL;
	}

	PyObject *module = PyImport_ImportModule("shapiq.tree.base");
	if (!module)
	{
		Py_DECREF(result);
		return NULL;
	}
	PyObject *tree_model_class = PyObject_GetAttrString(module, "TreeModel");
	Py_DECREF(module);
	if (!tree_model_class || !PyCallable_Check(tree_model_class))
	{
		Py_XDECREF(tree_model_class);
		Py_DECREF(result);
		PyErr_SetString(PyExc_RuntimeError, "Could not resolve callable TreeModel class.");
		return NULL;
	}

	for (size_t i = 0; i < n_trees; ++i)
	{
		const ParsedTreeArrays &tree = forest.trees[i];
		PyArrayObject *left_arr = numpy_array_from_vector<int64_t>(tree.left_children, NPY_INT64);
		PyArrayObject *right_arr = numpy_array_from_vector<int64_t>(tree.right_children, NPY_INT64);
		PyArrayObject *feature_arr = numpy_array_from_vector<int64_t>(tree.feature_ids, NPY_INT64);
		PyArrayObject *threshold_arr = float32_outputs ? numpy_float32_from_double_vector(tree.thresholds) : numpy_float64_from_double_vector(tree.thresholds);
		PyArrayObject *values_arr = float32_outputs ? numpy_float32_from_double_vector(tree.values) : numpy_float64_from_double_vector(tree.values);
		PyArrayObject *weight_arr = float32_outputs ? numpy_float32_from_double_vector(tree.node_sample_weights) : numpy_float64_from_double_vector(tree.node_sample_weights);
		if (!left_arr || !right_arr || !feature_arr || !threshold_arr || !values_arr || !weight_arr)
		{
			Py_XDECREF(left_arr);
			Py_XDECREF(right_arr);
			Py_XDECREF(feature_arr);
			Py_XDECREF(threshold_arr);
			Py_XDECREF(values_arr);
			Py_XDECREF(weight_arr);
			Py_DECREF(tree_model_class);
			Py_DECREF(result);
			return NULL;
		}

		npy_intp dims[1] = {PyArray_DIM(left_arr, 0)};
		PyArrayObject *children_missing_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
		if (!children_missing_arr)
		{
			Py_DECREF(left_arr);
			Py_DECREF(right_arr);
			Py_DECREF(feature_arr);
			Py_DECREF(threshold_arr);
			Py_DECREF(values_arr);
			Py_DECREF(weight_arr);
			Py_DECREF(tree_model_class);
			Py_DECREF(result);
			return NULL;
		}

		int64_t *missing_ptr = static_cast<int64_t *>(PyArray_DATA(children_missing_arr));
		const int64_t *default_ptr = tree.default_children.data();
		for (npy_intp j = 0; j < dims[0]; ++j)
		{
			missing_ptr[j] = default_ptr[j];
		}

		PyObject *kwargs = PyDict_New();
		if (!kwargs)
		{
			Py_DECREF(left_arr);
			Py_DECREF(right_arr);
			Py_DECREF(feature_arr);
			Py_DECREF(threshold_arr);
			Py_DECREF(values_arr);
			Py_DECREF(weight_arr);
			Py_DECREF(children_missing_arr);
			Py_DECREF(tree_model_class);
			Py_DECREF(result);
			return NULL;
		}

		if (
			PyDict_SetItemString(kwargs, "children_left", reinterpret_cast<PyObject *>(left_arr)) < 0 ||
			PyDict_SetItemString(kwargs, "children_right", reinterpret_cast<PyObject *>(right_arr)) < 0 ||
			PyDict_SetItemString(kwargs, "children_missing", reinterpret_cast<PyObject *>(children_missing_arr)) < 0 ||
			PyDict_SetItemString(kwargs, "features", reinterpret_cast<PyObject *>(feature_arr)) < 0 ||
			PyDict_SetItemString(kwargs, "thresholds", reinterpret_cast<PyObject *>(threshold_arr)) < 0 ||
			PyDict_SetItemString(kwargs, "values", reinterpret_cast<PyObject *>(values_arr)) < 0 ||
			PyDict_SetItemString(kwargs, "node_sample_weight", reinterpret_cast<PyObject *>(weight_arr)) < 0)
		{
			Py_DECREF(kwargs);
			Py_DECREF(left_arr);
			Py_DECREF(right_arr);
			Py_DECREF(feature_arr);
			Py_DECREF(threshold_arr);
			Py_DECREF(values_arr);
			Py_DECREF(weight_arr);
			Py_DECREF(children_missing_arr);
			Py_DECREF(tree_model_class);
			Py_DECREF(result);
			return NULL;
		}
		PyObject *decision_type = PyUnicode_FromString(decision_type_cstr);
		if (!decision_type)
		{
			Py_DECREF(kwargs);
			Py_DECREF(left_arr);
			Py_DECREF(right_arr);
			Py_DECREF(feature_arr);
			Py_DECREF(threshold_arr);
			Py_DECREF(values_arr);
			Py_DECREF(weight_arr);
			Py_DECREF(children_missing_arr);
			Py_DECREF(tree_model_class);
			Py_DECREF(result);
			return NULL;
		}
		if (PyDict_SetItemString(kwargs, "decision_type", decision_type) < 0)
		{
			Py_DECREF(decision_type);
			Py_DECREF(kwargs);
			Py_DECREF(left_arr);
			Py_DECREF(right_arr);
			Py_DECREF(feature_arr);
			Py_DECREF(threshold_arr);
			Py_DECREF(values_arr);
			Py_DECREF(weight_arr);
			Py_DECREF(children_missing_arr);
			Py_DECREF(tree_model_class);
			Py_DECREF(result);
			return NULL;
		}
		Py_DECREF(decision_type);

		PyObject *empty_args = PyTuple_New(0);
		if (!empty_args)
		{
			Py_DECREF(kwargs);
			Py_DECREF(left_arr);
			Py_DECREF(right_arr);
			Py_DECREF(feature_arr);
			Py_DECREF(threshold_arr);
			Py_DECREF(values_arr);
			Py_DECREF(weight_arr);
			Py_DECREF(children_missing_arr);
			Py_DECREF(tree_model_class);
			Py_DECREF(result);
			return NULL;
		}
		PyObject *tree_obj = PyObject_Call(tree_model_class, empty_args, kwargs);
		Py_DECREF(empty_args);
		Py_DECREF(kwargs);
		Py_DECREF(left_arr);
		Py_DECREF(right_arr);
		Py_DECREF(feature_arr);
		Py_DECREF(threshold_arr);
		Py_DECREF(values_arr);
		Py_DECREF(weight_arr);
		Py_DECREF(children_missing_arr);

		if (!tree_obj)
		{
			Py_DECREF(tree_model_class);
			Py_DECREF(result);
			return NULL;
		}

		PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), tree_obj);
	}

	Py_DECREF(tree_model_class);
	return result;
}

static int64_t comb_int(int64_t n, int64_t k)
{
	if (k < 0 || k > n)
		return 0;
	if (k > n - k)
		k = n - k;
	int64_t result = 1;
	for (int64_t i = 1; i <= k; ++i)
	{
		result = result * (n - k + i) / i;
	}
	return result;
}

static std::vector<std::vector<std::vector<int64_t>>> parse_subset_update_positions(
	PyObject *subset_updates_pos_store,
	int64_t max_interaction,
	int64_t n_features)
{
	std::vector<std::vector<std::vector<int64_t>>> updates(static_cast<size_t>(max_interaction + 1));
	for (int64_t order = 1; order <= max_interaction; ++order)
	{
		PyObject *order_key = PyLong_FromLongLong(order);
		if (!order_key)
			throw std::runtime_error("Could not allocate interaction-order key.");
		PyObject *order_dict = PyDict_GetItemWithError(subset_updates_pos_store, order_key);
		Py_DECREF(order_key);
		if (!order_dict)
		{
			if (PyErr_Occurred())
				throw std::runtime_error("Could not read subset update positions.");
			throw std::runtime_error("Missing subset update positions for an interaction order.");
		}
		if (!PyDict_Check(order_dict))
			throw std::runtime_error("Subset update positions for an interaction order must be a dict.");

		updates[static_cast<size_t>(order)].resize(static_cast<size_t>(n_features));
		for (int64_t feature = 0; feature < n_features; ++feature)
		{
			PyObject *feature_key = PyLong_FromLongLong(feature);
			if (!feature_key)
				throw std::runtime_error("Could not allocate feature key.");
			PyObject *positions = PyDict_GetItemWithError(order_dict, feature_key);
			Py_DECREF(feature_key);
			if (!positions)
			{
				if (PyErr_Occurred())
					throw std::runtime_error("Could not read feature subset update positions.");
				throw std::runtime_error("Missing subset update positions for a feature.");
			}
			PyArrayObject *positions_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(positions, NPY_INT64, NPY_ARRAY_IN_ARRAY));
			if (!positions_array)
				throw std::runtime_error("Subset update positions must be integer arrays.");
			int64_t *data = static_cast<int64_t *>(PyArray_DATA(positions_array));
			npy_intp size = PyArray_SIZE(positions_array);
			updates[static_cast<size_t>(order)][static_cast<size_t>(feature)].assign(data, data + size);
			Py_DECREF(positions_array);
		}
	}
	return updates;
}

static PyObject *create_edge_tree_arrays(PyObject *self, PyObject *args)
{
	(void)self;
	PyObject *children_left_obj;
	PyObject *children_right_obj;
	PyObject *features_obj;
	PyObject *node_sample_weight_obj;
	PyObject *values_obj;
	PyObject *subset_updates_pos_store;
	int n_nodes_int;
	int n_features_int;
	int max_interaction_int;

	if (!PyArg_ParseTuple(
			args,
			"OOOOOiiiO",
			&children_left_obj,
			&children_right_obj,
			&features_obj,
			&node_sample_weight_obj,
			&values_obj,
			&n_nodes_int,
			&n_features_int,
			&max_interaction_int,
			&subset_updates_pos_store))
	{
		return NULL;
	}

	PyArrayObject *children_left_arr = NULL;
	PyArrayObject *children_right_arr = NULL;
	PyArrayObject *features_arr = NULL;
	PyArrayObject *node_sample_weight_arr = NULL;
	PyArrayObject *values_arr = NULL;
	PyObject *parents_obj = NULL;
	PyObject *ancestors_obj = NULL;
	PyObject *ancestor_nodes_obj = NULL;
	PyObject *p_e_values_obj = NULL;
	PyObject *p_e_storages_obj = NULL;
	PyObject *split_weights_obj = NULL;
	PyObject *empty_predictions_obj = NULL;
	PyObject *edge_heights_obj = NULL;
	PyObject *last_feature_node_obj = NULL;
	PyObject *interaction_height_dict = NULL;

	try
	{
		if (!PyDict_Check(subset_updates_pos_store))
			throw std::runtime_error("subset_updates_pos_store must be a dict.");
		const int64_t n_nodes = n_nodes_int;
		const int64_t n_features = n_features_int;
		const int64_t max_interaction = max_interaction_int;
		if (n_nodes <= 0 || n_features < 0 || max_interaction <= 0)
			throw std::runtime_error("n_nodes and max_interaction must be positive; n_features must be non-negative.");

		children_left_arr = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(children_left_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
		children_right_arr = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(children_right_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
		features_arr = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(features_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
		node_sample_weight_arr = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(node_sample_weight_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
		values_arr = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
		if (!children_left_arr || !children_right_arr || !features_arr || !node_sample_weight_arr || !values_arr)
			throw std::runtime_error("Could not convert EdgeTree inputs to contiguous arrays.");
		if (PyArray_SIZE(children_left_arr) < n_nodes || PyArray_SIZE(children_right_arr) < n_nodes || PyArray_SIZE(features_arr) < n_nodes || PyArray_SIZE(node_sample_weight_arr) < n_nodes || PyArray_SIZE(values_arr) < n_nodes)
			throw std::runtime_error("EdgeTree input arrays are shorter than n_nodes.");

		const int64_t *children_left = static_cast<int64_t *>(PyArray_DATA(children_left_arr));
		const int64_t *children_right = static_cast<int64_t *>(PyArray_DATA(children_right_arr));
		const int64_t *features = static_cast<int64_t *>(PyArray_DATA(features_arr));
		const double *node_sample_weight = static_cast<double *>(PyArray_DATA(node_sample_weight_arr));
		const double *values = static_cast<double *>(PyArray_DATA(values_arr));
		std::vector<std::vector<std::vector<int64_t>>> subset_updates = parse_subset_update_positions(subset_updates_pos_store, max_interaction, n_features);

		npy_intp one_dim[1] = {static_cast<npy_intp>(n_nodes)};
		npy_intp storage_dims[2] = {static_cast<npy_intp>(n_nodes), static_cast<npy_intp>(n_features)};
		parents_obj = PyArray_SimpleNew(1, one_dim, NPY_INT64);
		ancestors_obj = PyArray_SimpleNew(1, one_dim, NPY_INT64);
		ancestor_nodes_obj = PyArray_SimpleNew(2, storage_dims, NPY_INT64);
		p_e_values_obj = PyArray_SimpleNew(1, one_dim, NPY_DOUBLE);
		p_e_storages_obj = PyArray_SimpleNew(2, storage_dims, NPY_DOUBLE);
		split_weights_obj = PyArray_SimpleNew(1, one_dim, NPY_DOUBLE);
		empty_predictions_obj = PyArray_SimpleNew(1, one_dim, NPY_DOUBLE);
		edge_heights_obj = PyArray_SimpleNew(1, one_dim, NPY_INT64);
		last_feature_node_obj = PyArray_SimpleNew(1, one_dim, NPY_BOOL);
		interaction_height_dict = PyDict_New();
		if (!parents_obj || !ancestors_obj || !ancestor_nodes_obj || !p_e_values_obj || !p_e_storages_obj || !split_weights_obj || !empty_predictions_obj || !edge_heights_obj || !last_feature_node_obj || !interaction_height_dict)
			throw std::runtime_error("Could not allocate EdgeTree output arrays.");

		int64_t *parents = static_cast<int64_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(parents_obj)));
		int64_t *ancestors = static_cast<int64_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ancestors_obj)));
		int64_t *ancestor_nodes = static_cast<int64_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ancestor_nodes_obj)));
		double *p_e_values = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(p_e_values_obj)));
		double *p_e_storages = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(p_e_storages_obj)));
		double *split_weights = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(split_weights_obj)));
		double *empty_predictions = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(empty_predictions_obj)));
		int64_t *edge_heights = static_cast<int64_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(edge_heights_obj)));
		npy_bool *last_feature_node = static_cast<npy_bool *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(last_feature_node_obj)));

		std::fill(parents, parents + n_nodes, -1);
		std::fill(ancestors, ancestors + n_nodes, -1);
		std::fill(ancestor_nodes, ancestor_nodes + n_nodes * n_features, -1);
		std::fill(p_e_values, p_e_values + n_nodes, 1.0);
		std::fill(p_e_storages, p_e_storages + n_nodes * n_features, 1.0);
		std::fill(split_weights, split_weights + n_nodes, 1.0);
		std::fill(empty_predictions, empty_predictions + n_nodes, 0.0);
		std::fill(edge_heights, edge_heights + n_nodes, -1);
		std::fill(last_feature_node, last_feature_node + n_nodes, static_cast<npy_bool>(0));

		std::vector<int64_t *> interaction_heights(static_cast<size_t>(max_interaction + 1), nullptr);
		std::vector<int64_t> interaction_widths(static_cast<size_t>(max_interaction + 1), 0);
		for (int64_t order = 1; order <= max_interaction; ++order)
		{
			int64_t width = comb_int(n_features, order);
			interaction_widths[static_cast<size_t>(order)] = width;
			npy_intp dims[2] = {static_cast<npy_intp>(n_nodes), static_cast<npy_intp>(width)};
			PyObject *height_obj = PyArray_ZEROS(2, dims, NPY_INT64, 0);
			if (!height_obj)
				throw std::runtime_error("Could not allocate interaction height array.");
			PyObject *order_key = PyLong_FromLongLong(order);
			if (!order_key)
				throw std::runtime_error("Could not allocate interaction height key.");
			if (PyDict_SetItem(interaction_height_dict, order_key, height_obj) < 0)
			{
				Py_DECREF(order_key);
				Py_DECREF(height_obj);
				throw std::runtime_error("Could not store interaction height array.");
			}
			Py_DECREF(order_key);
			interaction_heights[static_cast<size_t>(order)] = static_cast<int64_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(height_obj)));
			Py_DECREF(height_obj);
		}

		auto build_result = [&](int64_t max_depth_value) -> PyObject *
		{
			PyObject *result_tuple = PyTuple_New(11);
			if (!result_tuple)
				throw std::runtime_error("Could not allocate EdgeTree result tuple.");
			PyObject *max_depth_obj = PyLong_FromLongLong(max_depth_value);
			if (!max_depth_obj)
			{
				Py_DECREF(result_tuple);
				throw std::runtime_error("Could not allocate EdgeTree max_depth value.");
			}
			PyTuple_SET_ITEM(result_tuple, 0, parents_obj);
			PyTuple_SET_ITEM(result_tuple, 1, ancestors_obj);
			PyTuple_SET_ITEM(result_tuple, 2, ancestor_nodes_obj);
			PyTuple_SET_ITEM(result_tuple, 3, p_e_values_obj);
			PyTuple_SET_ITEM(result_tuple, 4, p_e_storages_obj);
			PyTuple_SET_ITEM(result_tuple, 5, split_weights_obj);
			PyTuple_SET_ITEM(result_tuple, 6, empty_predictions_obj);
			PyTuple_SET_ITEM(result_tuple, 7, edge_heights_obj);
			PyTuple_SET_ITEM(result_tuple, 8, max_depth_obj);
			PyTuple_SET_ITEM(result_tuple, 9, last_feature_node_obj);
			PyTuple_SET_ITEM(result_tuple, 10, interaction_height_dict);
			parents_obj = ancestors_obj = ancestor_nodes_obj = p_e_values_obj = p_e_storages_obj = NULL;
			split_weights_obj = empty_predictions_obj = edge_heights_obj = last_feature_node_obj = interaction_height_dict = NULL;
			return result_tuple;
		};

		if (children_left[0] == -1)
		{
			empty_predictions[0] = values[0];
			edge_heights[0] = 0;
			std::fill(last_feature_node, last_feature_node + n_nodes, static_cast<npy_bool>(1));
			PyObject *result = build_result(0);
			Py_DECREF(children_left_arr);
			Py_DECREF(children_right_arr);
			Py_DECREF(features_arr);
			Py_DECREF(node_sample_weight_arr);
			Py_DECREF(values_arr);
			return result;
		}

		int64_t max_depth = 0;
		auto recursive_search = [&](auto &&self, int64_t node_id, int64_t depth, double prod_weight, std::vector<int64_t> &seen_features) -> int64_t
		{
			max_depth = std::max(max_depth, depth);
			int64_t left_child = children_left[node_id];
			int64_t right_child = children_right[node_id];
			bool is_leaf = left_child == -1;
			if (!is_leaf)
			{
				parents[left_child] = node_id;
				parents[right_child] = node_id;
			}

			if (node_id == 0)
			{
				int64_t edge_heights_left = self(self, left_child, depth + 1, prod_weight, seen_features);
				int64_t edge_heights_right = self(self, right_child, depth + 1, prod_weight, seen_features);
				edge_heights[node_id] = std::max(edge_heights_left, edge_heights_right);
				return edge_heights[node_id];
			}

			std::memcpy(ancestor_nodes + node_id * n_features, seen_features.data(), static_cast<size_t>(n_features) * sizeof(int64_t));
			int64_t parent_id = parents[node_id];
			int64_t feature_id = features[parent_id];
			if (feature_id < 0 || feature_id >= n_features)
				throw std::runtime_error("EdgeTree feature id is outside the dense feature range.");

			last_feature_node[node_id] = 1;
			double weight = node_sample_weight[node_id] / node_sample_weight[parent_id];
			split_weights[node_id] = weight;
			prod_weight *= weight;
			double p_e = 1.0 / weight;

			for (int64_t order = 1; order <= max_interaction; ++order)
			{
				int64_t width = interaction_widths[static_cast<size_t>(order)];
				int64_t *height = interaction_heights[static_cast<size_t>(order)];
				std::memcpy(height + node_id * width, height + parent_id * width, static_cast<size_t>(width) * sizeof(int64_t));
			}

			if (seen_features[feature_id] > -1)
			{
				int64_t ancestor_id = seen_features[feature_id];
				ancestors[node_id] = ancestor_id;
				last_feature_node[ancestor_id] = 0;
				p_e *= p_e_values[ancestor_id];
			}
			else
			{
				for (int64_t order = 1; order <= max_interaction; ++order)
				{
					int64_t width = interaction_widths[static_cast<size_t>(order)];
					int64_t *height = interaction_heights[static_cast<size_t>(order)];
					for (int64_t index : subset_updates[static_cast<size_t>(order)][static_cast<size_t>(feature_id)])
					{
						if (index < 0 || index >= width)
							throw std::runtime_error("Subset update position is outside interaction height width.");
						height[node_id * width + index] += 1;
					}
				}
			}

			p_e_values[node_id] = p_e;
			std::memcpy(p_e_storages + node_id * n_features, p_e_storages + parent_id * n_features, static_cast<size_t>(n_features) * sizeof(double));
			p_e_storages[node_id * n_features + feature_id] = p_e;
			int64_t previous_seen_feature = seen_features[feature_id];
			seen_features[feature_id] = node_id;

			if (!is_leaf)
			{
				int64_t edge_heights_left = self(self, left_child, depth + 1, prod_weight, seen_features);
				int64_t edge_heights_right = self(self, right_child, depth + 1, prod_weight, seen_features);
				edge_heights[node_id] = std::max(edge_heights_left, edge_heights_right);
			}
			else
			{
				int64_t seen_count = 0;
				for (int64_t seen : seen_features)
				{
					if (seen > -1)
						seen_count++;
				}
				edge_heights[node_id] = seen_count;
				empty_predictions[node_id] = prod_weight * values[node_id];
			}
			seen_features[feature_id] = previous_seen_feature;
			return edge_heights[node_id];
		};

		std::vector<int64_t> seen_features(static_cast<size_t>(n_features), -1);
		(void)recursive_search(recursive_search, 0, 0, 1.0, seen_features);

		PyObject *result = build_result(max_depth);
		Py_DECREF(children_left_arr);
		Py_DECREF(children_right_arr);
		Py_DECREF(features_arr);
		Py_DECREF(node_sample_weight_arr);
		Py_DECREF(values_arr);
		return result;
	}
	catch (const std::exception &exc)
	{
		Py_XDECREF(children_left_arr);
		Py_XDECREF(children_right_arr);
		Py_XDECREF(features_arr);
		Py_XDECREF(node_sample_weight_arr);
		Py_XDECREF(values_arr);
		Py_XDECREF(parents_obj);
		Py_XDECREF(ancestors_obj);
		Py_XDECREF(ancestor_nodes_obj);
		Py_XDECREF(p_e_values_obj);
		Py_XDECREF(p_e_storages_obj);
		Py_XDECREF(split_weights_obj);
		Py_XDECREF(empty_predictions_obj);
		Py_XDECREF(edge_heights_obj);
		Py_XDECREF(last_feature_node_obj);
		Py_XDECREF(interaction_height_dict);
		PyErr_SetString(PyExc_RuntimeError, exc.what());
		return NULL;
	}
}


static PyObject *parse_xgboost_ubjson(PyObject *self, PyObject *args)
{
	(void)self;
	Py_buffer ubjson_buffer;
	int class_label = -1;
	double margin_base_score = 0.0;
	ubjson_buffer.buf = NULL;

	if (!PyArg_ParseTuple(args, "y*id", &ubjson_buffer, &class_label, &margin_base_score))
	{
		return NULL;
	}
	try
	{
		ParsedForest forest = parse_xgboost_ubjson_to_forest(
			static_cast<const uint8_t *>(ubjson_buffer.buf),
			static_cast<size_t>(ubjson_buffer.len),
			class_label,
			margin_base_score);

		PyObject *node_ids = forest_field_to_pylist<int64_t>(forest, &ParsedTreeArrays::node_ids, NPY_INT64);
		PyObject *feature_ids = forest_field_to_pylist<int64_t>(forest, &ParsedTreeArrays::feature_ids, NPY_INT64);
		PyObject *thresholds = forest_double_field_to_float32_pylist(forest, &ParsedTreeArrays::thresholds);
		PyObject *values = forest_double_field_to_float32_pylist(forest, &ParsedTreeArrays::values);
		PyObject *left_children = forest_field_to_pylist<int64_t>(forest, &ParsedTreeArrays::left_children, NPY_INT64);
		PyObject *right_children = forest_field_to_pylist<int64_t>(forest, &ParsedTreeArrays::right_children, NPY_INT64);
		PyObject *default_children = forest_field_to_pylist<int64_t>(forest, &ParsedTreeArrays::default_children, NPY_INT64);
		PyObject *node_sample_weights = forest_double_field_to_float32_pylist(forest, &ParsedTreeArrays::node_sample_weights);

		if (!node_ids || !feature_ids || !thresholds || !values || !left_children || !right_children || !default_children || !node_sample_weights)
		{
			Py_XDECREF(node_ids);
			Py_XDECREF(feature_ids);
			Py_XDECREF(thresholds);
			Py_XDECREF(values);
			Py_XDECREF(left_children);
			Py_XDECREF(right_children);
			Py_XDECREF(default_children);
			Py_XDECREF(node_sample_weights);
			PyBuffer_Release(&ubjson_buffer);
			return NULL;
		}

		PyObject *result = PyTuple_Pack(
			8,
			node_ids,
			feature_ids,
			thresholds,
			values,
			left_children,
			right_children,
			default_children,
			node_sample_weights);

		Py_DECREF(node_ids);
		Py_DECREF(feature_ids);
		Py_DECREF(thresholds);
		Py_DECREF(values);
		Py_DECREF(left_children);
		Py_DECREF(right_children);
		Py_DECREF(default_children);
		Py_DECREF(node_sample_weights);

		PyBuffer_Release(&ubjson_buffer);
		return result;
	}
	catch (const std::exception &exc)
	{
		PyBuffer_Release(&ubjson_buffer);
		PyErr_SetString(PyExc_RuntimeError, exc.what());
		return NULL;
	}
}

static PyObject *parse_xgboost_ubjson_treemodels(PyObject *self, PyObject *args)
{
	(void)self;
	Py_buffer ubjson_buffer;
	int class_label = -1;
	double margin_base_score = 0.0;
	ubjson_buffer.buf = NULL;

	if (!PyArg_ParseTuple(args, "y*id", &ubjson_buffer, &class_label, &margin_base_score))
	{
		return NULL;
	}

	try
	{
		ParsedForest forest = parse_xgboost_ubjson_to_forest(
			static_cast<const uint8_t *>(ubjson_buffer.buf),
			static_cast<size_t>(ubjson_buffer.len),
			class_label,
			margin_base_score);
		PyObject *result = forest_to_treemodel_list(forest, "<", true);
		if (!result)
		{
			PyBuffer_Release(&ubjson_buffer);
			return NULL;
		}
		PyBuffer_Release(&ubjson_buffer);
		return result;
	}
	catch (const std::exception &exc)
	{
		PyBuffer_Release(&ubjson_buffer);
		PyErr_SetString(PyExc_RuntimeError, exc.what());
		return NULL;
	}
}

static PyObject *parse_lightgbm_string_treemodels(PyObject *self, PyObject *args)
{
	(void)self;
	Py_buffer model_string_buffer;
	int class_label = -1;
	model_string_buffer.buf = NULL;

	if (!PyArg_ParseTuple(args, "y*i", &model_string_buffer, &class_label))
	{
		return NULL;
	}

	try
	{
		ParsedForest forest = parse_lightgbm_text_to_forest(
			static_cast<const char *>(model_string_buffer.buf),
			static_cast<size_t>(model_string_buffer.len),
			class_label);
		PyObject *result = forest_to_treemodel_list(forest, "<=", false);
		if (!result)
		{
			PyBuffer_Release(&model_string_buffer);
			return NULL;
		}
		PyBuffer_Release(&model_string_buffer);
		return result;
	}
	catch (const std::exception &exc)
	{
		PyBuffer_Release(&model_string_buffer);
		PyErr_SetString(PyExc_RuntimeError, exc.what());
		return NULL;
	}
}

static PyObject *parse_catboost_json_treemodels(PyObject *self, PyObject *args)
{
	(void)self;
	Py_buffer json_buffer;
	int class_label = -1;
	json_buffer.buf = NULL;

	if (!PyArg_ParseTuple(args, "y*i", &json_buffer, &class_label))
	{
		return NULL;
	}

	try
	{
		ParsedForest forest = parse_catboost_json_to_forest(static_cast<const char *>(json_buffer.buf), static_cast<size_t>(json_buffer.len), class_label);
		PyObject *result = forest_to_treemodel_list(forest, "<=", false);
		PyBuffer_Release(&json_buffer);
		return result;
	}
	catch (const std::exception &exc)
	{
		PyBuffer_Release(&json_buffer);
		PyErr_SetString(PyExc_RuntimeError, exc.what());
		return NULL;
	}
}
