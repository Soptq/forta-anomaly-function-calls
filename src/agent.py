from forta_agent import Finding, FindingType, FindingSeverity, TransactionEvent, BlockEvent, Label, EntityType

import warnings
import cachetools
import time
from pyod.models.ecod import ECOD
import numpy as np
import src.config as config

cached_contract_selectors = {}
cached_function_calls = {}
detector = ECOD(contamination=1e-5)
np.seterr(all="ignore")
warnings.filterwarnings('ignore')


def parse_traces(transaction_event: TransactionEvent):
    findings = []
    print(f"Parsing {len(transaction_event.traces)} traces for transaction {transaction_event.transaction.hash}")

    function_calls = []
    if len(transaction_event.traces) > 0:
        # get all traces with `call` type
        for trace in transaction_event.traces:
            if trace.error is not None:
                continue
            # deal with suicided contract
            if trace.type.lower() == 'suicide':
                suicided_contract = trace.action.address
                if suicided_contract in cached_contract_selectors:
                    del cached_contract_selectors[suicided_contract]
                if suicided_contract in cached_function_calls:
                    del cached_function_calls[suicided_contract]
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

            function_calls.append((caller, contract, trace.action.input))
    else:
        # no traces, this is a regular transaction
        if transaction_event.transaction.data.lower() == '0x':
            # regular transfer, not a contract call
            return findings
        caller = transaction_event.transaction.from_
        contract = transaction_event.transaction.to
        _input = transaction_event.transaction.data
        function_calls.append((caller, contract, _input))

    for function_call in function_calls:
        caller, contract, _input = function_call
        selector = _input[:10]

        if contract not in cached_contract_selectors:
            cached_contract_selectors[contract] = []
        if selector not in cached_contract_selectors[contract]:
            cached_contract_selectors[contract].append(selector)

        selector_index = cached_contract_selectors[contract].index(selector)

        if contract not in cached_function_calls:
            cached_function_calls[contract] = cachetools.TTLCache(maxsize=2 ** 20, ttl=config.HISTORY_TTL)

        time_to_collect = max(cached_function_calls[contract].keys()) - min(cached_function_calls[contract].keys()) if len(
            cached_function_calls[contract]) > 0 else 0
        if len(cached_function_calls[contract]) >= config.MIN_RECORDS_TO_DETECT or time_to_collect >= config.MIN_TIME_TO_COLLECT_NS:
            print(
                f"[{len(cached_function_calls[contract])}][{time_to_collect // 10 ** 9}] Detecting anomaly for contract {contract} with selector {selector}")
            # construct dataset
            # 1. one-hot encoded function selectors
            # 2. percentage of calls from caller
            # 3. percentage of calls to this selector
            n_feats = len(cached_contract_selectors[contract]) + 2
            train, test = np.zeros((len(cached_function_calls[contract]), n_feats)), np.zeros((1, n_feats))
            total_sum_calls = [0 for _ in range(len(cached_contract_selectors[contract]))]
            user_sum_calls = {}
            user_func_sum_calls = [{} for _ in range(len(cached_contract_selectors[contract]))]
            # construct train features
            for i, record in enumerate(cached_function_calls[contract].values()):
                train_caller, train_selector_index = record
                if train_caller not in user_func_sum_calls[train_selector_index]:
                    user_func_sum_calls[train_selector_index][train_caller] = 0
                user_func_sum_calls[train_selector_index][train_caller] += 1
                if train_caller not in user_sum_calls:
                    user_sum_calls[train_caller] = 0
                user_sum_calls[train_caller] += 1
                total_sum_calls[train_selector_index] += 1
                # one-hot encoded function selectors
                train[i, train_selector_index] = 1
                # percentage of calls from caller
                train[i, len(cached_contract_selectors[contract])] = user_func_sum_calls[train_selector_index][
                                                                         train_caller] / total_sum_calls[
                                                                         train_selector_index]
                # percentage of calls to this function selector
                train[i, len(cached_contract_selectors[contract]) + 1] = user_func_sum_calls[train_selector_index][
                                                                             train_caller] / user_sum_calls[
                                                                             train_caller]
            # construct test features
            if caller not in user_func_sum_calls[selector_index]:
                user_func_sum_calls[selector_index][caller] = 0
            user_func_sum_calls[selector_index][caller] += 1
            if caller not in user_sum_calls:
                user_sum_calls[caller] = 0
            user_sum_calls[caller] += 1
            total_sum_calls[selector_index] += 1
            test[0, selector_index] = 1
            test[0, len(cached_contract_selectors[contract])] = user_func_sum_calls[selector_index][caller] / \
                                                                total_sum_calls[selector_index]
            test[0, len(cached_contract_selectors[contract]) + 1] = user_func_sum_calls[selector_index][caller] / \
                                                                    user_sum_calls[caller]

            # predict
            detector.fit(train)
            probs, confidence = detector.predict_proba(test, return_confidence=True)
            print(f"Anomaly score for {contract} with selector {selector}: {probs[0, 1]}:{confidence[0]}")
            findings.append((contract, selector, probs[0, 1], confidence[0]))
        else:
            print(
                f"[{len(cached_function_calls[contract])}][{time_to_collect // 10 ** 9}] Not enough records for contract {contract} with selector {selector}")

        cached_function_calls[contract][time.time_ns()] = (caller, selector_index)

    return findings


def handle_transaction(transaction_event: TransactionEvent):
    try:
        findings = []
        anomaly_detections = parse_traces(transaction_event)

        caller = transaction_event.transaction.from_
        for detection in anomaly_detections:
            contract_address, selector, anomaly_score, confidence = detection
            if anomaly_score > config.ANOMALY_THRESHOLD:
                findings.append(Finding({
                    'name': f'Abnormal Function Call Detected',
                    'description': f'Abnormal function call detected from {caller} to {contract_address} with selector {selector}',
                    'alert_id': 'ABNORMAL-FUNCTION-CALL-DETECTED-1',
                    'severity': FindingSeverity.Medium,
                    'type': FindingType.Suspicious,
                    'metadata': {
                        'contract_address': contract_address,
                        'caller': caller,
                        'function_selector': selector,
                        'anomaly_score': anomaly_score,
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


# def initialize():
#     # do some initialization on startup e.g. fetch data


def handle_block(block_event: BlockEvent):
    if block_event.block.number % 3600 == 0:  # 12 hours
        n_clean = 0
        for contract, ttl_cache in cached_function_calls.items():
            if len(ttl_cache) == 0:
                del cached_contract_selectors[contract]
                del cached_function_calls[contract]
                n_clean += 1
        print(f'cleaned {n_clean} contracts from cache')
    return []

# def handle_alert(alert_event):
#     findings = []
#     # detect some alert condition
#     return findings
