from forta_agent import Finding, FindingType, FindingSeverity, TransactionEvent, BlockEvent, Label, EntityType

import math
import warnings
import cachetools
import time
from pyod.models.ecod import ECOD
import numpy as np
import asyncio
import threading
from itertools import product

from joblib import cpu_count
from src.sharding import get_sharding_stats

SHARDING_BOT_ID = -1
SHARDING_CLUSTER_SIZE = -1
SHARDING_BASE = None
SHARDING_FILTER_PREFIX = None

start_time = 0
cached_contract_selectors_traces = {}
cached_function_calls_traces = {}
traces_models = {}
cached_event_selectors_logs = {}
cached_event_emits_logs = {}
logs_models = {}
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=DeprecationWarning)

is_training = False
background_training_task = None


def get_noise(scalar):
    return np.random.normal(0, scalar, 1)[0]


def assigned_to_me(contract_address):
    global SHARDING_FILTER_PREFIX, SHARDING_BASE, SHARDING_BOT_ID, SHARDING_CLUSTER_SIZE
    if SHARDING_FILTER_PREFIX is None or SHARDING_BASE is None:
        return True
    if contract_address.lower()[2: 2 + SHARDING_BASE] in SHARDING_FILTER_PREFIX:
        return True
    return False


def reset():
    global cached_function_calls_traces, cached_contract_selectors_traces, traces_models, cached_event_selectors_logs, cached_event_emits_logs, logs_models
    cached_contract_selectors_traces = {}
    cached_function_calls_traces = {}
    traces_models = {}
    cached_event_selectors_logs = {}
    cached_event_emits_logs = {}
    logs_models = {}


async def parse_traces(transaction_event: TransactionEvent, config):
    findings = []
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

            existed = False
            for _caller, _input in function_calls[contract]:
                if _caller == caller and _input == trace.action.input:
                    existed = True
                    break
            if not existed and assigned_to_me(contract):
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

        existed = False
        for _caller, _input_ in function_calls[contract]:
            if _caller == caller and _input_ == _input:
                existed = True
                break
        if not existed and assigned_to_me(contract):
            function_calls[contract].append((caller, _input))

    for contract, data in function_calls.items():
        callers, _inputs = [], []
        for datum in data:
            caller, _input = datum
            callers.append(caller)
            _inputs.append(_input)
        selectors = [inp[:10] for inp in _inputs]

        if contract not in cached_function_calls_traces:
            cached_function_calls_traces[contract] = {
                "cache": cachetools.TTLCache(maxsize=2 ** 20, ttl=config.HISTORY_TTL),
                "user_sum_calls": {},
                "user_func_sum_calls": {},
                "total_sum_calls": {},
            }

        if contract not in cached_contract_selectors_traces:
            cached_contract_selectors_traces[contract] = {"train": [], "test": []}
        for selector in selectors:
            if selector not in cached_contract_selectors_traces[contract]["test"] and selector not in \
                    cached_contract_selectors_traces[contract]["train"]:
                cached_contract_selectors_traces[contract]["train"].append(selector)
                cached_function_calls_traces[contract]["user_func_sum_calls"][selector] = {}
                cached_function_calls_traces[contract]["total_sum_calls"][selector] = 0

        time_bot_started = time.time_ns() - start_time
        time_to_collect = max(cached_function_calls_traces[contract]["cache"].keys()) - min(
            cached_function_calls_traces[contract]["cache"].keys()) if len(
            cached_function_calls_traces[contract]["cache"]) > 0 else 0
        if (contract in traces_models and not traces_models[contract]["training"]) and ((
                                                                                                time_bot_started >= config.COLD_START_TIME and len(
                                                                                                cached_function_calls_traces[
                                                                                                    contract][
                                                                                                    "cache"]) >= config.MIN_RECORDS_TO_DETECT) \
                                                                                        or (
                                                                                                time_to_collect >= config.MIN_TIME_TO_COLLECT_NS and len(
                                                                                            cached_function_calls_traces[
                                                                                                contract][
                                                                                                "cache"]) >= config.MIN_RECORDS_TO_DETECT_FOR_MIN_TIME)):
            print(
                f"[{len(cached_function_calls_traces[contract]['cache'])}][{time_to_collect // 10 ** 9}] Detecting anomaly for contract {contract} with selectors {selectors}, batch size {len(data)}")
            n_feats = len(cached_contract_selectors_traces[contract]["test"]) + 3
            test_dataset = np.zeros((len(data), n_feats))

            # construct test features
            for i, (test_caller, test_selector) in enumerate(zip(callers, selectors)):
                if test_selector in cached_contract_selectors_traces[contract]["test"]:
                    test_selector_index = cached_contract_selectors_traces[contract]["test"].index(test_selector)
                else:
                    test_selector_index = len(cached_contract_selectors_traces[contract]["test"])
                test_dataset[i, test_selector_index] = 1 + get_noise(config.NOISE_SCALAR)

                if test_selector in cached_function_calls_traces[contract]["user_func_sum_calls"]:
                    if test_caller in cached_function_calls_traces[contract]["user_func_sum_calls"][test_selector]:
                        user_func_sum_calls = \
                        cached_function_calls_traces[contract]["user_func_sum_calls"][test_selector][test_caller]
                    else:
                        user_func_sum_calls = 0
                else:
                    user_func_sum_calls = 0

                if test_selector in cached_function_calls_traces[contract]["total_sum_calls"]:
                    total_sum_calls = cached_function_calls_traces[contract]["total_sum_calls"][test_selector]
                else:
                    total_sum_calls = 0

                if test_caller in cached_function_calls_traces[contract]["user_sum_calls"]:
                    user_sum_calls = cached_function_calls_traces[contract]["user_sum_calls"][test_caller]
                else:
                    user_sum_calls = 0

                test_dataset[i, n_feats - 2] = (user_func_sum_calls + 1) / (total_sum_calls + 1) + get_noise(
                    config.NOISE_SCALAR)
                test_dataset[i, n_feats - 1] = (user_func_sum_calls + 1) / (user_sum_calls + 1) + get_noise(
                    config.NOISE_SCALAR)

            probs, confidences = traces_models[contract]["model"].predict_proba(test_dataset, return_confidence=True)
            print(f"Anomaly score for {contract} with selector {selectors}: {probs}:{confidences}")
            for selector, prob, confidence in zip(selectors, probs, confidences):
                findings.append((contract, selector, prob[1], confidence))
        else:
            print(
                f"[{len(cached_function_calls_traces[contract]['cache'])}][{time_to_collect // 10 ** 9}] Not enough records for contract {contract} with selectors {selectors}, batch size {len(data)}")

        for i, (caller, selector) in enumerate(zip(callers, selectors)):
            cached_function_calls_traces[contract]["cache"][time.time_ns()] = (caller, selector)
            if caller not in cached_function_calls_traces[contract]["user_func_sum_calls"][selector]:
                cached_function_calls_traces[contract]["user_func_sum_calls"][selector][caller] = 0
            cached_function_calls_traces[contract]["user_func_sum_calls"][selector][caller] += 1
            if caller not in cached_function_calls_traces[contract]["user_sum_calls"]:
                cached_function_calls_traces[contract]["user_sum_calls"][caller] = 0
            cached_function_calls_traces[contract]["user_sum_calls"][caller] += 1
            cached_function_calls_traces[contract]["total_sum_calls"][selector] += 1
    return findings


async def parse_logs(transaction_event: TransactionEvent, config):
    findings = []
    events_emit = {}
    caller = transaction_event.transaction.from_
    for log in transaction_event.logs:
        if len(log.topics) == 0:
            continue

        contract = log.address
        event_selector = log.topics[0]

        if contract not in events_emit:
            events_emit[contract] = []

        existed = False
        for (_caller, _event_selector) in events_emit[contract]:
            if _caller == caller and _event_selector == event_selector:
                existed = True
                break
        if not existed and assigned_to_me(contract):
            events_emit[contract].append((caller, event_selector))

    for contract, data in events_emit.items():
        callers, event_selectors = [], []
        for datum in data:
            caller, event_selector = datum
            callers.append(caller)
            event_selectors.append(event_selector)

        if contract not in cached_event_emits_logs:
            cached_event_emits_logs[contract] = {
                "cache": cachetools.TTLCache(maxsize=2 ** 20, ttl=config.HISTORY_TTL),
                "user_sum_calls": {},
                "user_func_sum_calls": {},
                "total_sum_calls": {}
            }

        if contract not in cached_event_selectors_logs:
            cached_event_selectors_logs[contract] = {"train": [], "test": []}
        for selector in event_selectors:
            if selector not in cached_event_selectors_logs[contract]["test"] and selector not in \
                    cached_event_selectors_logs[contract]["train"]:
                cached_event_selectors_logs[contract]["train"].append(selector)
                cached_event_emits_logs[contract]["user_func_sum_calls"][selector] = {}
                cached_event_emits_logs[contract]["total_sum_calls"][selector] = 0

        time_bot_started = time.time_ns() - start_time
        time_to_collect = max(cached_event_emits_logs[contract]["cache"].keys()) - min(
            cached_event_emits_logs[contract]["cache"].keys()) if len(
            cached_event_emits_logs[contract]["cache"]) > 0 else 0
        if (contract in logs_models and not logs_models[contract]["training"]) and ((
                                                                                            time_bot_started >= config.COLD_START_TIME and len(
                                                                                            cached_event_emits_logs[
                                                                                                contract]) >= config.MIN_RECORDS_TO_DETECT) or (
                                                                                            time_to_collect >= config.MIN_TIME_TO_COLLECT_NS and len(
                                                                                        cached_event_emits_logs[
                                                                                            contract]) >= config.MIN_RECORDS_TO_DETECT_FOR_MIN_TIME)):
            print(
                f"[{len(cached_event_emits_logs[contract]['cache'])}][{time_to_collect // 10 ** 9}] Detecting anomaly for contract {contract} with event selectors {event_selectors}, batch size {len(data)}")
            n_feats = len(cached_event_selectors_logs[contract]["test"]) + 3
            test_dataset = np.zeros((len(data), n_feats))

            # construct test features
            for i, (test_caller, test_selector) in enumerate(zip(callers, event_selectors)):
                if test_selector in cached_event_selectors_logs[contract]["test"]:
                    test_selector_index = cached_event_selectors_logs[contract]["test"].index(test_selector)
                else:
                    test_selector_index = len(cached_event_selectors_logs[contract]["test"])
                test_dataset[i, test_selector_index] = 1 + get_noise(config.NOISE_SCALAR)

                if test_selector in cached_event_emits_logs[contract]["user_func_sum_calls"]:
                    if test_caller in cached_event_emits_logs[contract]["user_func_sum_calls"][test_selector]:
                        user_func_sum_calls = cached_event_emits_logs[contract]["user_func_sum_calls"][test_selector][
                            test_caller]
                    else:
                        user_func_sum_calls = 0
                else:
                    user_func_sum_calls = 0

                if test_selector in cached_event_emits_logs[contract]["total_sum_calls"]:
                    total_sum_calls = cached_event_emits_logs[contract]["total_sum_calls"][test_selector]
                else:
                    total_sum_calls = 0

                if test_caller in cached_event_emits_logs[contract]["user_sum_calls"]:
                    user_sum_calls = cached_event_emits_logs[contract]["user_sum_calls"][test_caller]
                else:
                    user_sum_calls = 0

                test_dataset[i, n_feats - 2] = (user_func_sum_calls + 1) / (total_sum_calls + 1) + get_noise(
                    config.NOISE_SCALAR)
                test_dataset[i, n_feats - 1] = (user_func_sum_calls + 1) / (user_sum_calls + 1) + get_noise(
                    config.NOISE_SCALAR)

            probs, confidences = logs_models[contract]["model"].predict_proba(test_dataset, return_confidence=True)
            print(f"Anomaly score for {contract} with selector {event_selectors}: {probs}:{confidences}")
            for selector, prob, confidence in zip(event_selectors, probs, confidences):
                findings.append((contract, selector, prob[1], confidence))
        else:
            print(
                f"[{len(cached_event_emits_logs[contract]['cache'])}][{time_to_collect // 10 ** 9}] Not enough records for contract {contract} with event selectors {event_selectors}, batch size {len(data)}")

        for i, (caller, selector) in enumerate(zip(callers, event_selectors)):
            cached_event_emits_logs[contract]["cache"][time.time_ns()] = (caller, selector)
            if caller not in cached_event_emits_logs[contract]["user_func_sum_calls"][selector]:
                cached_event_emits_logs[contract]["user_func_sum_calls"][selector][caller] = 0
            cached_event_emits_logs[contract]["user_func_sum_calls"][selector][caller] += 1
            if caller not in cached_event_emits_logs[contract]["user_sum_calls"]:
                cached_event_emits_logs[contract]["user_sum_calls"][caller] = 0
            cached_event_emits_logs[contract]["user_sum_calls"][caller] += 1
            cached_event_emits_logs[contract]["total_sum_calls"][selector] += 1

    return findings


async def detect_traces_alerts(caller, anomaly_detections_traces, config):
    findings = []
    max_anomaly = (None, None, 0., None)
    for detection in anomaly_detections_traces:
        contract_address, selector, anomaly_score, confidence = detection
        if anomaly_score > config.ANOMALY_THRESHOLD and confidence > config.CONFIDENCE_THRESHOLD:
            if anomaly_score > max_anomaly[2]:
                max_anomaly = (contract_address, selector, anomaly_score, confidence)
    if max_anomaly[2] > config.ANOMALY_THRESHOLD and max_anomaly[3] > config.CONFIDENCE_THRESHOLD:
        contract_address, selector, anomaly_score, confidence = max_anomaly
        findings.append(Finding({
            'name': f'Abnormal Function Call Detected',
            'description': f'Abnormal function call detected from {caller} to {contract_address} with selector {selector}, anomaly score {max(config.LOWEST_ANOMALY_SCORE, 1.0 - anomaly_score)}',
            'alert_id': 'ABNORMAL-FUNCTION-CALL-DETECTED-1',
            'severity': FindingSeverity.Medium,
            'type': FindingType.Suspicious,
            'metadata': {
                'contract_address': contract_address,
                'caller': caller,
                'function_selector': selector,
                'anomaly_score': max(config.LOWEST_ANOMALY_SCORE, 1.0 - anomaly_score),  # 0 is the most abnormal
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


async def detect_logs_alerts(caller, anomaly_detections_logs, config):
    findings = []
    max_anomaly = (None, None, 0., None)
    for detection in anomaly_detections_logs:
        contract_address, selector, anomaly_score, confidence = detection
        if anomaly_score > config.ANOMALY_THRESHOLD and confidence > config.CONFIDENCE_THRESHOLD:
            if anomaly_score > max_anomaly[2]:
                max_anomaly = (contract_address, selector, anomaly_score, confidence)
    if max_anomaly[2] > config.ANOMALY_THRESHOLD and max_anomaly[3] > config.CONFIDENCE_THRESHOLD:
        contract_address, selector, anomaly_score, confidence = max_anomaly
        findings.append(Finding({
            'name': f'Abnormal Emitted Event Detected',
            'description': f'Abnormal emitted event detected from {caller} to {contract_address} with topic {selector}, anomaly score {max(config.LOWEST_ANOMALY_SCORE, 1.0 - anomaly_score)}',
            'alert_id': 'ABNORMAL-EMITTED-EVENT-DETECTED-1',
            'severity': FindingSeverity.Medium,
            'type': FindingType.Suspicious,
            'metadata': {
                'contract_address': contract_address,
                'caller': caller,
                'event_topic': selector,
                'anomaly_score': max(config.LOWEST_ANOMALY_SCORE, 1.0 - anomaly_score),  # 0 is the most abnormal
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


async def provide_handle_transaction(transaction_event: TransactionEvent, config):
    findings = []
    [anomaly_detections_traces, anomaly_detections_logs] = await asyncio.gather(
        parse_traces(transaction_event, config),
        parse_logs(transaction_event, config)
    )

    caller = transaction_event.transaction.from_
    [traces_alerts, logs_alerts] = await asyncio.gather(
        detect_traces_alerts(caller, anomaly_detections_traces, config),
        detect_logs_alerts(caller, anomaly_detections_logs, config)
    )

    findings.extend(traces_alerts)
    findings.extend(logs_alerts)

    return findings


def handle_transaction_sync(transaction_event: TransactionEvent, config):
    return asyncio.run(provide_handle_transaction(transaction_event, config))


def handle_transaction(transaction_event: TransactionEvent):
    import src.config as config
    return asyncio.run(provide_handle_transaction(transaction_event, config))


def update_sharding_stats(block_event: BlockEvent):
    global SHARDING_BOT_ID, SHARDING_CLUSTER_SIZE, SHARDING_BASE, SHARDING_FILTER_PREFIX
    try:
        SHARDING_BOT_ID, SHARDING_CLUSTER_SIZE = get_sharding_stats(int(block_event.network))
    except:
        SHARDING_BOT_ID, SHARDING_CLUSTER_SIZE = 0, 1
    print(f'sharding stats: {SHARDING_BOT_ID}, {SHARDING_CLUSTER_SIZE}')

    SHARDING_BASE = math.ceil(math.log2(SHARDING_CLUSTER_SIZE) / 4) if SHARDING_CLUSTER_SIZE > 1 else 1
    open_set = list(map(''.join, product('0123456789abcdef', repeat=SHARDING_BASE)))
    cluster_set = np.array_split(open_set, SHARDING_CLUSTER_SIZE)
    SHARDING_FILTER_PREFIX = cluster_set[SHARDING_BOT_ID]


def initialize():
    # do some initialization on startup e.g. fetch data
    global start_time, SHARDING_BOT_ID, SHARDING_CLUSTER_SIZE, SHARDING_BASE, SHARDING_FILTER_PREFIX
    start_time = time.time_ns()


def provide_handle_block(block_event: BlockEvent, config):
    global is_training, SHARDING_BOT_ID, SHARDING_CLUSTER_SIZE
    is_training = True

    update_sharding_stats(block_event)

    n_clean = 0
    will_train_contract_traces, will_train_contract_logs = [], []
    for contract, data in cached_function_calls_traces.items():
        if len(data["cache"]) == 0:
            del cached_contract_selectors_traces[contract]
            del cached_function_calls_traces[contract]
            del traces_models[contract]
            n_clean += 1
        else:
            if len(cached_contract_selectors_traces[contract]["train"]) > 0:
                will_train_contract_traces.append(contract)

    for contract, data in cached_event_emits_logs.items():
        if len(data["cache"]) == 0:
            del cached_event_selectors_logs[contract]
            del cached_event_emits_logs[contract]
            del logs_models[contract]
            n_clean += 1
        else:
            if len(cached_event_selectors_logs[contract]["train"]) > 0:
                will_train_contract_logs.append(contract)
    print(f'cleaned {n_clean} contracts from cache')

    for contract in will_train_contract_traces:
        print(f'training traces model for contract {contract}')
        traces_models[contract] = {"model": None, "training": True}

        cached_contract_selectors_traces[contract]["test"].extend(
            cached_contract_selectors_traces[contract]["train"])
        cached_contract_selectors_traces[contract]["train"] = []

        n_feats = len(cached_contract_selectors_traces[contract]["test"]) + 3
        train_dataset = np.zeros((len(cached_function_calls_traces[contract]["cache"]), n_feats))

        for i, record in enumerate(cached_function_calls_traces[contract]["cache"].values()):
            caller, selector = record
            selector_index = cached_contract_selectors_traces[contract]["test"].index(selector)
            train_dataset[i, selector_index] = 1 + get_noise(config.NOISE_SCALAR)
            train_dataset[i, n_feats - 2] = cached_function_calls_traces[contract]["user_func_sum_calls"][selector][
                                                caller] / cached_function_calls_traces[contract]["total_sum_calls"][
                                                selector] + get_noise(config.NOISE_SCALAR)
            train_dataset[i, n_feats - 1] = cached_function_calls_traces[contract]["user_func_sum_calls"][selector][
                                                caller] / cached_function_calls_traces[contract]["user_sum_calls"][
                                                caller] + get_noise(config.NOISE_SCALAR)

        traces_models[contract]["model"] = ECOD(contamination=config.NOISE_SCALAR,
                                                n_jobs=min(cpu_count(), train_dataset.shape[1]))
        try:
            traces_models[contract]["model"].fit(train_dataset)
            traces_models[contract]["training"] = False
        except:
            del traces_models[contract]
            continue

    for contract in will_train_contract_logs:
        print(f'training logs model for contract {contract}')
        logs_models[contract] = {"model": None, "training": True}

        cached_event_selectors_logs[contract]["test"].extend(
            cached_event_selectors_logs[contract]["train"])
        cached_event_selectors_logs[contract]["train"] = []

        n_feats = len(cached_event_selectors_logs[contract]["test"]) + 3
        train_dataset = np.zeros((len(cached_event_emits_logs[contract]["cache"]), n_feats))

        for i, record in enumerate(cached_event_emits_logs[contract]["cache"].values()):
            caller, selector = record
            selector_index = cached_event_selectors_logs[contract]["test"].index(selector)
            train_dataset[i, selector_index] = 1 + get_noise(config.NOISE_SCALAR)
            train_dataset[i, n_feats - 2] = cached_event_emits_logs[contract]["user_func_sum_calls"][selector][caller] / \
                                            cached_event_emits_logs[contract]["total_sum_calls"][selector] + get_noise(
                config.NOISE_SCALAR)
            train_dataset[i, n_feats - 1] = cached_event_emits_logs[contract]["user_func_sum_calls"][selector][caller] / \
                                            cached_event_emits_logs[contract]["user_sum_calls"][caller] + get_noise(
                config.NOISE_SCALAR)

        logs_models[contract]["model"] = ECOD(contamination=config.NOISE_SCALAR,
                                              n_jobs=min(cpu_count(), train_dataset.shape[1]))
        try:
            logs_models[contract]["model"].fit(train_dataset)
            logs_models[contract]["training"] = False
        except:
            del logs_models[contract]
            continue

    is_training = False


def handle_block(block_event: BlockEvent):
    global is_training, background_training_task, SHARDING_BOT_ID, SHARDING_CLUSTER_SIZE
    import src.config as config

    if SHARDING_BOT_ID == -1 or SHARDING_CLUSTER_SIZE == -1:
        update_sharding_stats(block_event)

    if is_training:
        return []
    if block_event.block.number % config.MAINTAIN_INTERVAL_BLK[int(block_event.network)] == 0:
        background_training_task = threading.Thread(target=provide_handle_block, args=(block_event, config))
        background_training_task.start()
    return []

# def handle_alert(alert_event):
#     findings = []
#     # detect some alert condition
#     return findings
