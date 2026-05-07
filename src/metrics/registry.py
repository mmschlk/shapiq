from .distance_metrics import MSEMetric, NormalizedMSEMetric

METRICS = {
    "mse": MSEMetric(),
    "normalized_mse":NormalizedMSEMetric
}