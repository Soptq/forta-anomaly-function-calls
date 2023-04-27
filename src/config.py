HISTORY_TTL = 60 * 60 * 24 * 30  # 30 days
MIN_RECORDS_TO_DETECT = 2 ** 14  # there must be at least 16384 function call records to detect a possible anomaly.
MIN_TIME_TO_COLLECT_NS = 3 * 60 * 60 * 10 ** 9  # 180 minutes
MIN_RECORDS_TO_DETECT_FOR_MIN_TIME = 2 ** 6
ANOMALY_THRESHOLD = 0.9  # the threshold to determine whether a function call is an anomaly.
NOISE_SCALAR = 1e-10
COLD_START_TIME = 72 * 60 * 60 * 10 ** 9  # 72 hours
CONFIDENCE_THRESHOLD = 0.8
MAINTAIN_INTERVAL_BLK = {
    1: 300 * 3,
    10: 1800 * 3,
    56: 1200 * 3,
    137: 1800 * 3,
    250: 3600 * 3,
    42161: 240 * 3,
    43114: 180 * 3,
}
