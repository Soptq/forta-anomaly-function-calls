from unittest.mock import patch
from forta_agent import create_transaction_event, create_block_event, FindingSeverity, EntityType
import agent


class TestAnomalyDetectionAgent:
    def test_1_anomaly_function_call(self):
        agent.reset()
        import config
        config.MIN_RECORDS_TO_DETECT = 8
        config.MAINTAIN_INTERVAL_BLK = 1
        tx_event = create_transaction_event(
            {
                "transaction": {
                    "hash": "0",
                    "to": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e1111",
                    "from": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e0000",
                    "value": "0",
                    "data": "0x1234567800000000"
                },
                "block": {"number": 1},
                "receipt": {"logs": []},
            }
        )

        anomaly_tx_event = create_transaction_event(
            {
                "transaction": {
                    "hash": "0",
                    "to": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e1111",
                    "from": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e0000",
                    "value": "0",
                    "data": "0x0000000000000000"
                },
                "block": {"number": 1},
                "receipt": {"logs": []},
            }
        )

        block_event = create_block_event({"block": {"number": 1}})

        for tx_event in [tx_event] * 10:
            findings = agent.provide_handle_transaction(tx_event, config)
            assert (
                    len(findings) == 0
            ), "this should have not triggered a finding"

        agent.provide_handle_block(block_event, config)
        agent.provide_handle_block(block_event, config)
        findings = agent.provide_handle_transaction(anomaly_tx_event, config)
        assert (
            len(findings) != 0
        ), "this should have triggered a finding"

    def test_2_anomaly_function_call(self):
        agent.reset()
        import config
        config.MIN_RECORDS_TO_DETECT = 16
        config.MAINTAIN_INTERVAL_BLK = 1
        tx_event_1 = create_transaction_event(
            {
                "transaction": {
                    "hash": "0",
                    "to": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e1111",
                    "from": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e0000",
                    "value": "0",
                    "data": "0x1111111100000000"
                },
                "block": {"number": 1},
                "receipt": {"logs": []},
            }
        )

        tx_event_2 = create_transaction_event(
            {
                "transaction": {
                    "hash": "0",
                    "to": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e1111",
                    "from": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e0000",
                    "value": "0",
                    "data": "0x2222222200000000"
                },
                "block": {"number": 1},
                "receipt": {"logs": []},
            }
        )

        anomaly_tx_event = create_transaction_event(
            {
                "transaction": {
                    "hash": "0",
                    "to": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e1111",
                    "from": "0x4e5b2e1dc63f6b91cb6cd759936495434c7e0000",
                    "value": "0",
                    "data": "0x0000000000000000"
                },
                "block": {"number": 1},
                "receipt": {"logs": []},
            }
        )

        block_event = create_block_event({"block": {"number": 1}})
        agent.provide_handle_block(block_event, config)

        for tx_event in [tx_event_1, tx_event_2] * 8:
            findings = agent.provide_handle_transaction(tx_event, config)
            print(findings)
            assert (
                    len(findings) == 0
            ), "this should have not triggered a finding"

        agent.provide_handle_block(block_event, config)

        findings = agent.provide_handle_transaction(anomaly_tx_event, config)
        assert (
            len(findings) != 0
        ), "this should have triggered a finding"