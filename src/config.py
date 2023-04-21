HISTORY_TTL = 60 * 60 * 24 * 30  # 30 days
MIN_RECORDS_TO_DETECT = 2 ** 12  # there must be at least 4096 function call records to detect a possible anomaly.
MIN_TIME_TO_COLLECT_NS = 60 * 60 * 10 ** 9  # 60 minutes
MIN_RECORDS_TO_DETECT_FOR_MIN_TIME = 2 ** 6
ANOMALY_THRESHOLD = 0.9  # the threshold to determine whether a function call is an anomaly.
NOISE_SCALAR = 1e-10
COLD_START_TIME = 72 * 60 * 60 * 10 ** 9  # 72 hours
