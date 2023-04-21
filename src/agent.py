from forta_agent import Finding, FindingType, FindingSeverity, TransactionEvent, BlockEvent, Label, EntityType

import warnings
import cachetools
import time
from pyod.models.ecod import ECOD
import numpy as np

start_time = 0
cached_contract_selectors_traces = {}
cached_function_calls_traces = {}
cached_event_selectors_logs = {}
cached_event_emits_logs = {}
warnings.filterwarnings("error")


def get_noise(scalar):
    return np.random.normal(0, scalar, 1)[0]


def reset():
    global cached_function_calls_traces, cached_contract_selectors_traces, cached_event_selectors_logs, cached_event_emits_logs
    cached_contract_selectors_traces = {}
    cached_function_calls_traces = {}
    cached_event_selectors_logs = {}
    cached_event_emits_logs = {}


def parse_traces(transaction_event: TransactionEvent, config):
    findings = []
    print(f"Parsing {len(transaction_event.traces)} traces for transaction {transaction_event.transaction.hash}")

    function_calls = {}
    if len(transaction_event.traces) > 0:
        # get all traces with `call` type
        for trace in transaction_event.traces:
            if trace.error is not None and trace.error != "":
                continue
            # deal with suicided contract
            if trace.type.lower() == 'suicide':
                suicided_contract = trace.action.address
                if suicided_contract in cached_contract_selectors_traces:
                    del cached_contract_selectors_traces[suicided_contract]
                if suicided_contract in cached_function_calls_traces:
                    del cached_function_calls_traces[suicided_contract]
                continue

            if trace.type.lower() != 'call':
                continue
            if trace.action.input.lower() == '0x':
                # regular transfer, not a contract call
                continue

            if trace.action.call_type.lower() == 'call':
                caller = trace.action.from_
                contract = trace.action.to
            elif trace.action.call_type.lower() == 'callcode':
                # proxy call, there will be another call trace with call_type `call` or `staticcall`
                # so we will ignore it
                continue
            elif trace.action.call_type.lower() == 'delegatecall':
                # proxy call, there will be another call trace with call_type `call` or `staticcall` to the proxy contract
                # so we will ignore it
                continue
            elif trace.action.call_type.lower() == 'staticcall':
                # this is a read-only call, should not cause any state changes
                continue
            else:
                raise Exception(f'Unknown call type {trace.action.call_type}')

            if contract not in function_calls:
                function_calls[contract] = []
            function_calls[contract].append((caller, trace.action.input))
    else:
        # no traces, this is a regular transaction
        if transaction_event.transaction.data.lower() == '0x':
            # regular transfer, not a contract call
            return findings
        caller = transaction_event.transaction.from_
        contract = transaction_event.transaction.to
        _input = transaction_event.transaction.data
        if contract not in function_calls:
            function_calls[contract] = []
        function_calls[contract].append((caller, _input))

    for contract, data in function_calls.items():
        callers, _inputs = [], []
        for datum in data:
            caller, _input = datum
            callers.append(caller)
            _inputs.append(_input)
        selectors = [inp[:10] for inp in _inputs]

        if contract not in cached_contract_selectors_traces:
            cached_contract_selectors_traces[contract] = []
        for selector in selectors:
            if selector not in cached_contract_selectors_traces[contract]:
                cached_contract_selectors_traces[contract].append(selector)

        selector_indexes = [cached_contract_selectors_traces[contract].index(selector) for selector in selectors]

        if contract not in cached_function_calls_traces:
            cached_function_calls_traces[contract] = cachetools.TTLCache(maxsize=2 ** 20, ttl=config.HISTORY_TTL)

        time_bot_started = time.time_ns() - start_time
        time_to_collect = max(cached_function_calls_traces[contract].keys()) - min(
            cached_function_calls_traces[contract].keys()) if len(
            cached_function_calls_traces[contract]) > 0 else 0
        if time_bot_started >= config.COLD_START_TIME and len(cached_function_calls_traces[contract]) >= config.MIN_RECORDS_TO_DETECT or (
                time_to_collect >= config.MIN_TIME_TO_COLLECT_NS and len(
            cached_function_calls_traces[contract]) >= config.MIN_RECORDS_TO_DETECT_FOR_MIN_TIME):
            print(
                f"[{len(cached_function_calls_traces[contract])}][{time_to_collect // 10 ** 9}] Detecting anomaly for contract {contract} with selectors {selectors}, batch size {len(data)}")
            # construct dataset
            # 1. one-hot encoded function selectors
            # 2. percentage of calls from caller
            # 3. percentage of calls to this selector
            n_feats = len(cached_contract_selectors_traces[contract]) + 2
            train, test = np.zeros((len(cached_function_calls_traces[contract]), n_feats)), np.zeros((len(data), n_feats))
            total_sum_calls = [0 for _ in range(len(cached_contract_selectors_traces[contract]))]
            user_sum_calls = {}
            user_func_sum_calls = [{} for _ in range(len(cached_contract_selectors_traces[contract]))]

            for i, record in enumerate(cached_function_calls_traces[contract].values()):
                train_caller, train_selector_index = record
                if train_caller not in user_func_sum_calls[train_selector_index]:
                    user_func_sum_calls[train_selector_index][train_caller] = 0
                user_func_sum_calls[train_selector_index][train_caller] += 1
                if train_caller not in user_sum_calls:
                    user_sum_calls[train_caller] = 0
                user_sum_calls[train_caller] += 1
                total_sum_calls[train_selector_index] += 1
            for i, (test_caller, test_selector_index) in enumerate(zip(callers, selector_indexes)):
                if test_caller not in user_func_sum_calls[test_selector_index]:
                    user_func_sum_calls[test_selector_index][test_caller] = 0
                user_func_sum_calls[test_selector_index][test_caller] += 1
                if test_caller not in user_sum_calls:
                    user_sum_calls[test_caller] = 0
                user_sum_calls[test_caller] += 1
                total_sum_calls[test_selector_index] += 1

            # construct train features
            for i, record in enumerate(cached_function_calls_traces[contract].values()):
                train_caller, train_selector_index = record
                # one-hot encoded function selectors
                train[i, train_selector_index] = 1 + get_noise(config.NOISE_SCALAR)
                # percentage of calls from caller
                train[i, len(cached_contract_selectors_traces[contract])] = user_func_sum_calls[train_selector_index][
                                                                         train_caller] / total_sum_calls[
                                                                         train_selector_index] + get_noise(
                    config.NOISE_SCALAR)
                # percentage of calls to this function selector
                train[i, len(cached_contract_selectors_traces[contract]) + 1] = user_func_sum_calls[train_selector_index][
                                                                             train_caller] / user_sum_calls[
                                                                             train_caller] + get_noise(
                    config.NOISE_SCALAR)

            # construct test features
            for i, (test_caller, test_selector_index) in enumerate(zip(callers, selector_indexes)):
                test[i, test_selector_index] = 1 + get_noise(config.NOISE_SCALAR)
                test[i, len(cached_contract_selectors_traces[contract])] = user_func_sum_calls[test_selector_index][
                                                                        test_caller] / \
                                                                    total_sum_calls[test_selector_index] + get_noise(
                    config.NOISE_SCALAR)
                test[i, len(cached_contract_selectors_traces[contract]) + 1] = user_func_sum_calls[test_selector_index][
                                                                            test_caller] / \
                                                                        user_sum_calls[test_caller] + get_noise(
                    config.NOISE_SCALAR)

            # predict
            detector = ECOD(contamination=config.NOISE_SCALAR, n_jobs=1)
            try:
                detector.fit(train)
            except:
                continue
            probs, confidences = detector.predict_proba(test, return_confidence=True)
            print(f"Anomaly score for {contract} with selector {selectors}: {probs}:{confidences}")
            for selector, prob, confidence in zip(selectors, probs, confidences):
                findings.append((contract, selector, prob[1], confidence))
        else:
            print(
                f"[{len(cached_function_calls_traces[contract])}][{time_to_collect // 10 ** 9}] Not enough records for contract {contract} with selector {selector}, batch size {len(data)}")

        for i, (caller, selector_index) in enumerate(zip(callers, selector_indexes)):
            cached_function_calls_traces[contract][time.time_ns()] = (caller, selector_index)

    return findings


def parse_logs(transaction_event: TransactionEvent, config):
    findings = []
    print(f"Parsing {len(transaction_event.logs)} logs for transaction {transaction_event.transaction.hash}")

    events_emit = {}
    caller = transaction_event.transaction.from_
    for log in transaction_event.logs:
        if len(log.topics) == 0:
            continue

        contract = log.address
        event_selector = log.topics[0]

        if contract not in events_emit:
            events_emit[contract] = []
        events_emit[contract].append((caller, event_selector))

    for contract, data in events_emit.items():
        callers, event_selectors = [], []
        for datum in data:
            caller, event_selector = datum
            callers.append(caller)
            event_selectors.append(event_selector)

        if contract not in cached_event_selectors_logs:
            cached_event_selectors_logs[contract] = []
        for selector in event_selectors:
            if selector not in cached_event_selectors_logs[contract]:
                cached_event_selectors_logs[contract].append(selector)

        selector_indexes = [cached_event_selectors_logs[contract].index(selector) for selector in event_selectors]

        if contract not in cached_event_emits_logs:
            cached_event_emits_logs[contract] = cachetools.TTLCache(maxsize=2 ** 20, ttl=config.HISTORY_TTL)

        time_bot_started = time.time_ns() - start_time
        time_to_collect = max(cached_event_emits_logs[contract].keys()) - min(
            cached_event_emits_logs[contract].keys()) if len(
            cached_event_emits_logs[contract]) > 0 else 0
        if time_bot_started >= config.COLD_START_TIME and len(cached_event_emits_logs[contract]) >= config.MIN_RECORDS_TO_DETECT or (
                time_to_collect >= config.MIN_TIME_TO_COLLECT_NS and len(
            cached_event_emits_logs[contract]) >= config.MIN_RECORDS_TO_DETECT_FOR_MIN_TIME):
            print(
                f"[{len(cached_event_emits_logs[contract])}][{time_to_collect // 10 ** 9}] Detecting anomaly for contract {contract} with event selectors {event_selectors}, batch size {len(data)}")
            # construct dataset
            # 1. one-hot encoded event selectors
            # 2. percentage of calls from caller
            # 3. percentage of calls to this selector
            n_feats = len(cached_event_selectors_logs[contract]) + 2
            train, test = np.zeros((len(cached_event_emits_logs[contract]), n_feats)), np.zeros((len(data), n_feats))
            total_sum_calls = [0 for _ in range(len(cached_event_selectors_logs[contract]))]
            user_sum_calls = {}
            user_func_sum_calls = [{} for _ in range(len(cached_event_selectors_logs[contract]))]

            for i, record in enumerate(cached_event_emits_logs[contract].values()):
                train_caller, train_selector_index = record
                if train_caller not in user_func_sum_calls[train_selector_index]:
                    user_func_sum_calls[train_selector_index][train_caller] = 0
                user_func_sum_calls[train_selector_index][train_caller] += 1
                if train_caller not in user_sum_calls:
                    user_sum_calls[train_caller] = 0
                user_sum_calls[train_caller] += 1
                total_sum_calls[train_selector_index] += 1
            for i, (test_caller, test_selector_index) in enumerate(zip(callers, selector_indexes)):
                if test_caller not in user_func_sum_calls[test_selector_index]:
                    user_func_sum_calls[test_selector_index][test_caller] = 0
                user_func_sum_calls[test_selector_index][test_caller] += 1
                if test_caller not in user_sum_calls:
                    user_sum_calls[test_caller] = 0
                user_sum_calls[test_caller] += 1
                total_sum_calls[test_selector_index] += 1

            # construct train features
            for i, record in enumerate(cached_event_emits_logs[contract].values()):
                train_caller, train_selector_index = record
                # one-hot encoded function selectors
                train[i, train_selector_index] = 1 + get_noise(config.NOISE_SCALAR)
                # percentage of calls from caller
                train[i, len(cached_event_selectors_logs[contract])] = user_func_sum_calls[train_selector_index][
                                                                         train_caller] / total_sum_calls[
                                                                         train_selector_index] + get_noise(
                    config.NOISE_SCALAR)
                # percentage of calls to this function selector
                train[i, len(cached_event_selectors_logs[contract]) + 1] = user_func_sum_calls[train_selector_index][
                                                                             train_caller] / user_sum_calls[
                                                                             train_caller] + get_noise(
                    config.NOISE_SCALAR)

            # construct test features
            for i, (test_caller, test_selector_index) in enumerate(zip(callers, selector_indexes)):
                test[i, test_selector_index] = 1 + get_noise(config.NOISE_SCALAR)
                test[i, len(cached_event_selectors_logs[contract])] = user_func_sum_calls[test_selector_index][
                                                                        test_caller] / \
                                                                    total_sum_calls[test_selector_index] + get_noise(
                    config.NOISE_SCALAR)
                test[i, len(cached_event_selectors_logs[contract]) + 1] = user_func_sum_calls[test_selector_index][
                                                                            test_caller] / \
                                                                        user_sum_calls[test_caller] + get_noise(
                    config.NOISE_SCALAR)

            # predict
            detector = ECOD(contamination=config.NOISE_SCALAR, n_jobs=1)
            try:
                detector.fit(train)
            except:
                continue
            probs, confidences = detector.predict_proba(test, return_confidence=True)
            print(f"Anomaly score for {contract} with selector {event_selectors}: {probs}:{confidences}")
            for selector, prob, confidence in zip(event_selectors, probs, confidences):
                findings.append((contract, selector, prob[1], confidence))
        else:
            print(
                f"[{len(cached_event_emits_logs[contract])}][{time_to_collect // 10 ** 9}] Not enough records for contract {contract} with event selector {event_selectors}, batch size {len(data)}")

        for i, (caller, selector_index) in enumerate(zip(callers, selector_indexes)):
            cached_event_emits_logs[contract][time.time_ns()] = (caller, selector_index)

    return findings


def provide_handle_transaction(transaction_event: TransactionEvent, config):
    try:
        findings = []
        anomaly_detections_traces = parse_traces(transaction_event, config)
        anomaly_detections_logs = parse_logs(transaction_event, config)

        caller = transaction_event.transaction.from_

        max_anomaly = (None, None, 0., None)
        for detection in anomaly_detections_traces:
            contract_address, selector, anomaly_score, confidence = detection
            if anomaly_score > max_anomaly[2]:
                max_anomaly = (contract_address, selector, anomaly_score, confidence)
        if max_anomaly[2] > config.ANOMALY_THRESHOLD:
            contract_address, selector, anomaly_score, confidence = max_anomaly
            findings.append(Finding({
                'name': f'Abnormal Function Call Detected',
                'description': f'Abnormal function call detected from {caller} to {contract_address} with selector {selector}, anomaly score {anomaly_score}',
                'alert_id': 'ABNORMAL-FUNCTION-CALL-DETECTED-1',
                'severity': FindingSeverity.Medium,
                'type': FindingType.Suspicious,
                'metadata': {
                    'contract_address': contract_address,
                    'caller': caller,
                    'function_selector': selector,
                    'anomaly_score': 1.0 - anomaly_score,  # 0 is the most abnormal
                    'confidence': confidence,
                },
                "labels": [
                    Label({
                        "entity": caller,
                        "entity_type": EntityType.Address,
                        "label": "attack",
                        "confidence": confidence
                    }),
                    Label({
                        "entity": contract_address,
                        "entity_type": EntityType.Address,
                        "label": "attack",
                        "confidence": confidence
                    }),
                ]
            }))

        max_anomaly = (None, None, 0., None)
        for detection in anomaly_detections_logs:
            contract_address, selector, anomaly_score, confidence = detection
            if anomaly_score > max_anomaly[2]:
                max_anomaly = (contract_address, selector, anomaly_score, confidence)
        if max_anomaly[2] > config.ANOMALY_THRESHOLD:
            contract_address, selector, anomaly_score, confidence = max_anomaly
            findings.append(Finding({
                'name': f'Abnormal Emitted Event Detected',
                'description': f'Abnormal emitted event detected from {caller} to {contract_address} with topic {selector}, anomaly score {anomaly_score}',
                'alert_id': 'ABNORMAL-EMITTED-EVENT-DETECTED-1',
                'severity': FindingSeverity.Medium,
                'type': FindingType.Suspicious,
                'metadata': {
                    'contract_address': contract_address,
                    'caller': caller,
                    'event_topic': selector,
                    'anomaly_score': 1.0 - anomaly_score,  # 0 is the most abnormal
                    'confidence': confidence,
                },
                "labels": [
                    Label({
                        "entity": caller,
                        "entity_type": EntityType.Address,
                        "label": "attack",
                        "confidence": confidence
                    }),
                    Label({
                        "entity": contract_address,
                        "entity_type": EntityType.Address,
                        "label": "attack",
                        "confidence": confidence
                    }),
                ]
            }))

        return findings
    except Exception as e:
        print(e)
        return []


def handle_transaction(transaction_event: TransactionEvent):
    import src.config as config
    return provide_handle_transaction(transaction_event, config)


def initialize():
    # do some initialization on startup e.g. fetch data
    global start_time
    start_time = time.time_ns()


def handle_block(block_event: BlockEvent):
    if block_event.block.number % 3600 == 0:  # 12 hours
        n_clean = 0
        for contract, ttl_cache in cached_function_calls_traces.items():
            if len(ttl_cache) == 0:
                del cached_contract_selectors_traces[contract]
                del cached_function_calls_traces[contract]
                n_clean += 1
        for contract, ttl_cache in cached_event_emits_logs.items():
            if len(ttl_cache) == 0:
                del cached_event_selectors_logs[contract]
                del cached_event_emits_logs[contract]
                n_clean += 1
        print(f'cleaned {n_clean} contracts from cache')
    return []

# def handle_alert(alert_event):
#     findings = []
#     # detect some alert condition
#     return findings
