HISTORY_TTL = 60 * 60 * 24 * 30  # 30 days
MIN_RECORDS_TO_DETECT = 2 ** 18  # there must be at least 2 ** 18 function call records to detect a possible anomaly.
MIN_TIME_TO_COLLECT_NS = 8 * 60 * 60 * 10 ** 9  # 8 hours
MIN_RECORDS_TO_DETECT_FOR_MIN_TIME = 2 ** 6
ANOMALY_THRESHOLD = 0.95  # the threshold to determine whether a function call is an anomaly.
NOISE_SCALAR = 1e-10
COLD_START_TIME = 120 * 60 * 60 * 10 ** 9  # 120 hours
CONFIDENCE_THRESHOLD = 0.9
MAINTAIN_INTERVAL_BLK = {
    1: 300,
    10: 1800,
    56: 1200,
    137: 1800,
    250: 3600,
    42161: 240,
    43114: 180,
}
LOWEST_ANOMALY_SCORE = 1e-3
