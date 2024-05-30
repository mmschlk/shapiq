⭐ Getting Started
==================

Explain a model with Shapley interaction values, e.g. the k-SII values.

..  code-block:: python

    import shapiq
    # load data
    X, y = shapiq.load_california_housing(to_numpy=True)
    # train a model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    # explain with k-SII interaction scores
    explainer = shapiq.TabularExplainer(
        model=model,
        data=X,
        index="k-SII",
        max_order=2
    )
    interaction_values = explainer.explain(X[0], budget=256)

    print(interaction_values)
    >> InteractionValues(
    >>    index=k-SII, max_order=2, min_order=0, estimated=False,
    >>    estimation_budget=256, n_players=8, baseline_value=0.86628,
    >>    Top 10 interactions:
    >>        (0,): 3.58948354047   # main effect for feature 0
    >>        (7,): 1.61175123142
    >>        (0, 1): 0.208496403   # interaction for features 0 & 1
    >>        (5,): 0.20069311333
    >>        (2,): 0.17536356571
    >>        (0, 5): -0.09740194
    >>        (0, 3): -0.12671954
    >>        (0, 6): -0.21245009
    >>        (6, 7): -0.34294075
    >>        (0, 7): -1.15889485
    >> )

    shapiq.network_plot(
        first_order_values=interaction_values.get_n_order_values(1),
        second_order_values=interaction_values.get_n_order_values(2)
    )
