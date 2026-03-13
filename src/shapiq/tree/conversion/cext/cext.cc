#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cstring>

#include "converter.cc"

static PyObject *parse_xgboost_ubjson(PyObject *self, PyObject *args);
static PyObject *parse_xgboost_ubjson_treemodels(PyObject *self, PyObject *args);
static PyObject *parse_lightgbm_string_treemodels(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
	{"parse_xgboost_ubjson", parse_xgboost_ubjson, METH_VARARGS, "Parse XGBoost UBJSON bytes and return tree-structure numpy-array lists."},
	{"parse_xgboost_ubjson_treemodels", parse_xgboost_ubjson_treemodels, METH_VARARGS, "Parse XGBoost UBJSON bytes and return a list of TreeModel objects."},
	{"parse_lightgbm_string_treemodels", parse_lightgbm_string_treemodels, METH_VARARGS, "Parse LightGBM model_to_string() bytes and return a list of TreeModel objects."},
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

// Cast a double vector to a float32 numpy array.  All float tree data
// (thresholds, values, node sample weights) is stored internally as double for
// arithmetic precision, then narrowed to float32 on output so that it matches
// the precision of XGBoost's internal DMatrix representation.
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

static PyObject *forest_to_treemodel_list(const ParsedForest &forest, bool xgboost_default_left_mask, const char *decision_type_cstr)
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
		PyArrayObject *threshold_arr = numpy_float32_from_double_vector(tree.thresholds);
		PyArrayObject *values_arr = numpy_float32_from_double_vector(tree.values);
		PyArrayObject *weight_arr = numpy_float32_from_double_vector(tree.node_sample_weights);
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
		if (xgboost_default_left_mask)
		{
			const int64_t *left_ptr = tree.left_children.data();
			const int64_t *right_ptr = tree.right_children.data();
			const int64_t *default_ptr = tree.default_children.data();
			for (npy_intp j = 0; j < dims[0]; ++j)
			{
				missing_ptr[j] = default_ptr[j] == 1 ? left_ptr[j] : right_ptr[j];
			}
		}
		else
		{
			const int64_t *default_ptr = tree.default_children.data();
			for (npy_intp j = 0; j < dims[0]; ++j)
			{
				missing_ptr[j] = default_ptr[j];
			}
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

		PyDict_SetItemString(kwargs, "children_left", reinterpret_cast<PyObject *>(left_arr));
		PyDict_SetItemString(kwargs, "children_right", reinterpret_cast<PyObject *>(right_arr));
		PyDict_SetItemString(kwargs, "children_missing", reinterpret_cast<PyObject *>(children_missing_arr));
		PyDict_SetItemString(kwargs, "features", reinterpret_cast<PyObject *>(feature_arr));
		PyDict_SetItemString(kwargs, "thresholds", reinterpret_cast<PyObject *>(threshold_arr));
		PyDict_SetItemString(kwargs, "values", reinterpret_cast<PyObject *>(values_arr));
		PyDict_SetItemString(kwargs, "node_sample_weight", reinterpret_cast<PyObject *>(weight_arr));
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
		PyDict_SetItemString(kwargs, "decision_type", decision_type);
		Py_DECREF(decision_type);

		PyObject *empty_args = PyTuple_New(0);
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

static PyObject *parse_xgboost_ubjson(PyObject *self, PyObject *args)
{
	(void)self;
	Py_buffer ubjson_buffer;
	int class_label = -1;
	ubjson_buffer.buf = NULL;

	if (!PyArg_ParseTuple(args, "y*i", &ubjson_buffer, &class_label))
	{
		return NULL;
	}
	try
	{
		ByteStream stream(static_cast<const uint8_t *>(ubjson_buffer.buf), static_cast<size_t>(ubjson_buffer.len));
		ParsedForest forest = stream.extractTreeStructure(class_label);

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
	ubjson_buffer.buf = NULL;

	if (!PyArg_ParseTuple(args, "y*i", &ubjson_buffer, &class_label))
	{
		return NULL;
	}

	try
	{
		ByteStream stream(static_cast<const uint8_t *>(ubjson_buffer.buf), static_cast<size_t>(ubjson_buffer.len));
		ParsedForest forest = stream.extractTreeStructure(class_label);
		PyObject *result = forest_to_treemodel_list(forest, true, "<");
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
		StringStream stream(static_cast<const char *>(model_string_buffer.buf), static_cast<size_t>(model_string_buffer.len));
		ParsedForest forest = stream.extractTreeStructure(class_label);
		PyObject *result = forest_to_treemodel_list(forest, false, "<=");
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
