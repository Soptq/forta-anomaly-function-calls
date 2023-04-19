# Forta Anomaly Function Calls Agent

## Description

A Forta Agent that detects anomalous function calls. Currently, the anomaly detection involves the following observations:

1. The function selector of the call.
2. Attributes of the caller like the frequency of calls and others.

The bot uses [ECOD](https://arxiv.org/abs/2201.00382) to detect anomalies. To increase the performance of the bot, function calls are batched to be processed.

## Supported Chains

- All chains that Forta supports.

## Alerts

- ABNORMAL-FUNCTION-CALL-DETECTED-1
  - Fired when a function call is suspicious using anomaly detection.
  - Severity is always set to "Medium".
  - Type is always set to "Suspicious".
  - Metadata:
    - `contract_address`: The address of the contract that was called.
    - `caller`: The address of the caller.
    - `function_selector`: The function signature of the call.
    - `anomaly_score`: The anomaly score of the call.
    - `confidence`: How consistently the model would make the same prediction if the training set was perturbed.

# Config

Configurable parameters are listed in `config.py`.
