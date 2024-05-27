⭐ Getting Started
==================

..  code-block:: python
    # train a model
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(x_train, y_train)

    # explain with k-SII interaction scores
    from shapiq import TabularExplainer

    explainer = TabularExplainer(
        model=model.predict,
        data=x_train,
        index="k-SII",
        max_order=2
    )
    interaction_values = explainer.explain(x, budget=2000)
    print(interaction_values)

    >> > InteractionValues(
        >> > index = k - SII, max_order = 2, min_order = 1, estimated = True, estimation_budget = 2000,
    >> > values = {
                >> > (0,): -91.0403,  # main effect for feature 0
    >> > (1,): 4.1264,  # main effect for feature 1
    >> > (2,): -0.4724,  # main effect for feature 2
    >> > ...
        >> > (0, 1): -0.8073,  # 2-way interaction for feature 0 and 1
    >> > (0, 2): 2.469,  # 2-way interaction for feature 0 and 2
    >> > ...
        >> > (10, 11): 0.4057  # 2-way interaction for feature 10 and 11
                        >> >}
    >> > )
