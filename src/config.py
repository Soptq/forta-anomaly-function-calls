HISTORY_TTL = 60 * 60 * 24 * 7  # 7 days
MIN_RECORDS_TO_DETECT = 1000  # there must be at least 100 function call records to detect a possible anomaly.
MIN_TIME_TO_COLLECT_NS = 30 * 60 * 10 ** 9  # 30 minutes
ANOMALY_THRESHOLD = 0.5  # the threshold to determine whether a function call is an anomaly.
