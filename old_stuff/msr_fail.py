# # estimate the Shapley values of the residual game values using MSR (SHAPIQ order 1)
# shapley_values_residual = np.zeros(self.n + 1)
#
# shapley_weights = np.zeros((self.n + 1, 2))
# for coalition_size in range(self.n + 1):
#     for intersection_size in range(2):
#         shapley_weights[
#             coalition_size, intersection_size
#         ] = self.shapley_weight(coalition_size - intersection_size)
#
# for i in range(self.n):
#     # get sampling parameters
#     coalitions_size = self._sampler.coalitions_size
#     set_i_binary = np.zeros(self.n, dtype=int)
#     set_i_binary[i] = 1
#     intersections_size = np.sum(coalitions_matrix * set_i_binary, axis=1)
#
#     weights = shapley_weights[coalitions_size, intersections_size]
#
#     shapley_values_residual[self.interaction_lookup[(i,)]] = np.sum(
#         weights * residual_values * self._sampler.sampling_adjustment_weights
#     )
#
# baseline_value = (
#     float(game_values[self._sampler.empty_coalition_index])
#     - shapley_tree.baseline_value
# )
# shapley_values_residual[self.interaction_lookup[()]] = baseline_value
#
# shapley_residuals = InteractionValues(
#     shapley_values_residual,
#     index=self.approximation_index,
#     n_players=self.n,
#     interaction_lookup=self.interaction_lookup,
#     min_order=self.min_order,
#     max_order=self.max_order,
#     baseline_value=baseline_value,
#     estimated=not budget >= 2**self.n,
#     estimation_budget=budget,
# )
