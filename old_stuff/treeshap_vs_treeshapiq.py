from shapiq import TreeExplainer
import copy

from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


if __name__ == "__main__":
    explanation_instance = 1
    #model =  RandomForestRegressor(random_state=42, n_estimators=20, max_depth=15)
    model = DecisionTreeRegressor(random_state=42)
    #model = XGBRegressor(random_state=42)
    background_data, y = make_regression(
        n_samples=1000, n_features=15, random_state=42
    )
    model.fit(background_data, y)

    x_explain_shap = background_data[explanation_instance].reshape(1, -1)
    x_explain_shapiq = background_data[explanation_instance]

    feature_perturbation = "interventional"

    if feature_perturbation == "tree_path_dependent":
        background_data = None
    if feature_perturbation == "interventional":
        background_data = copy.copy(background_data[2,:].reshape(1, -1))

    import shap
    model_copy = copy.deepcopy(model)
    explainer_shap = shap.TreeExplainer(model=model_copy, feature_perturbation=feature_perturbation, data=background_data)
    sv_shap = explainer_shap.shap_values(x_explain_shap)[0]

    # compute with shapiq
    explainer_shapiq = TreeExplainer(
        model=model,
        max_order=1,
        index="SV",
        feature_perturbation=feature_perturbation,
        background_data=background_data,
    )
    sv_shapiq = explainer_shapiq.explain(x=x_explain_shapiq)
    sv_shapiq_values = sv_shapiq.get_n_order_values(1)
    baseline_shapiq = sv_shapiq.baseline_value

    # print results
    print(sv_shap)
    print(sv_shapiq_values)
