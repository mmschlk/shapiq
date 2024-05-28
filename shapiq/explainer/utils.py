import re
import warnings

from .. import explainer


def get_explainers():
    return {"tabular": explainer.TabularExplainer, "tree": explainer.TreeExplainer}


def get_predict_function_and_model_type(model, model_class):
    _predict_function = None
    _model_type = "tabular"  # default

    if callable(model):
        _predict_function = predict_callable

    # sklearn
    if model_class in [
        "sklearn.tree.DecisionTreeRegressor",
        "sklearn.tree._classes.DecisionTreeRegressor",
        "sklearn.tree.DecisionTreeClassifier",
        "sklearn.tree._classes.DecisionTreeClassifier",
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn.ensemble._forest.RandomForestClassifier",
        "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn.ensemble._forest.ExtraTreesClassifier",
        "sklearn.ensemble.RandomForestRegressor",
        "sklearn.ensemble._forest.RandomForestRegressor",
    ]:
        _model_type = "tree"

    # lightgbm
    if model_class in [
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMRegressor",
        "lightgbm.sklearn.LGBMClassifier",
    ]:
        _model_type = "tree"

    # xgboost
    if model_class == "xgboost.core.Booster":
        _predict_function = predict_xgboost
    if model_class in [
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBRegressor",
        "xgboost.sklearn.XGBClassifier",
    ]:
        _model_type = "tree"

    # TODO: torch.Sequential

    # tensorflow
    if model_class in [
        "tensorflow.python.keras.engine.sequential.Sequential",
        "tensorflow.python.keras.engine.training.Model",
        "tensorflow.python.keras.engine.functional.Functional",
        "keras.engine.sequential.Sequential",
        "keras.engine.training.Model",
        "keras.engine.functional.Functional",
        "keras.src.models.sequential.Sequential",
    ]:
        _model_type = "tabular"
        if model.output_shape[1] == 1:
            _predict_function = predict_tf_single
        elif model.output_shape[1] == 2:
            _predict_function = predict_tf_binary
        else:
            _predict_function = predict_tf_first
            warnings.warn(
                "Tensorflow: Output shape of the model greater than 2. Explaining the 1st '0' class."
            )

    # default extraction (sklearn api)
    if _predict_function is None and hasattr(model, "predict_proba"):
        _predict_function = predict_proba_default
    elif _predict_function is None and hasattr(model, "predict"):
        _predict_function = predict_default
    elif isinstance(model, explainer.tree.TreeModel):  # test scenario
        _predict_function = model.compute_empty_prediction
        _model_type = "tree"
    elif _predict_function is None:
        raise TypeError(
            f"`model` is of unsupported type: {model_class}.\n"
            "Please, raise a new issue at https://github.com/mmschlk/shapiq/issues if you want this model type\n"
            "to be handled automatically by shapiq.Explainer. Otherwise, use one of the supported explainers:\n"
            f'{", ".join(print_classes_nicely(get_explainers()))}'
        )

    return _predict_function, _model_type


def predict_callable(m, d):
    return m(d)


def predict_default(m, d):
    return m.predict(d)


def predict_proba_default(m, d):
    return m.predict_proba(d)[:, 1]


def predict_xgboost(m, d):
    from xgboost import DMatrix

    return m.predict(DMatrix(d))


def predict_tf_single(m, d):
    return m.predict(d, verbose=0).reshape(
        -1,
    )


def predict_tf_binary(m, d):
    return m.predict(d, verbose=0)[:, 1]


def predict_tf_first(m, d):
    return m.predict(d, verbose=0)[:, 0]


def print_classes_nicely(obj):
    """
    Converts a list of classes into *user-readable* class names. I/O examples:
    [shapiq.explainer._base.Explainer] -> ['shapiq.Explainer']
    {'tree': shapiq.explainer.tree.explainer.TreeExplainer}  -> ['shapiq.TreeExplainer']
    {'tree': shapiq.TreeExplainer}  -> ['shapiq.TreeExplainer']
    """
    if isinstance(obj, dict):
        return [".".join([print_class(v).split(".")[i] for i in (0, -1)]) for _, v in obj.items()]
    elif isinstance(obj, list):
        return [".".join([print_class(v).split(".")[i] for i in (0, -1)]) for v in obj]


def print_class(obj):
    """
    Converts a class or class type into a *user-readable* class name. I/O examples:
    sklearn.ensemble._forest.RandomForestRegressor -> 'sklearn.ensemble._forest.RandomForestRegressor'
    type(sklearn.ensemble._forest.RandomForestRegressor) -> 'sklearn.ensemble._forest.RandomForestRegressor'
    shapiq.explainer.tree.explainer.TreeExplainer -> 'shapiq.explainer.tree.explainer.TreeExplainer'
    shapiq.TreeExplainer -> 'shapiq.explainer.tree.explainer.TreeExplainer'
    type(shapiq.TreeExplainer) -> 'shapiq.explainer.tree.explainer.TreeExplainer'
    """
    if isinstance(obj, type):
        return re.search("(?<=<class ').*(?='>)", str(obj))[0]
    else:
        return re.search("(?<=<class ').*(?='>)", str(type(obj)))[0]
