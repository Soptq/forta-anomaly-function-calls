import json
from forta_agent import Web3, fetch_jwt, decode_jwt
from src.constants import DISPATCH_CONTRACT_ADDRESS, SCANNER_POOL_CONTRACT_ADDRESS

polygon_provider = Web3.HTTPProvider('https://polygon-rpc.com')
web3 = Web3(polygon_provider)
dispatch_abi = json.load(open('src/abi/DispatchAbi.json'))
scanner_pool_abi = json.load(open('src/abi/ScannerPoolRegistryABI.json'))

dispatch = web3.eth.contract(address=Web3.toChecksumAddress(DISPATCH_CONTRACT_ADDRESS), abi=dispatch_abi)
scannerRegistry = web3.eth.contract(address=Web3.toChecksumAddress(SCANNER_POOL_CONTRACT_ADDRESS), abi=scanner_pool_abi)


def hex_to_int(h):
    return int(h, 0)


def int_to_address(i):
    h = hex(i).lower()[2:].zfill(40)
    return Web3.toChecksumAddress(f"0x{h}")


def get_bot_info():
    token = fetch_jwt({})
    payload = json.loads(decode_jwt(token).payload)
    bot_id = payload['bot-id']
    scanner = payload["sub"].lower()

    return bot_id, scanner


def get_scanners():
    bot_id, scanner = get_bot_info()
    n_scanners = dispatch.functions.numScannersFor(hex_to_int(bot_id)).call()

    scanners = []
    for i in range(n_scanners):
        scanner = dispatch.functions.scannerAt(hex_to_int(bot_id), i).call()
        scanners.append(scanner)

    chain_ids = []
    for scanner in scanners:
        scanner_state = scannerRegistry.functions.getScannerState(int_to_address(scanner)).call()
        chain_id = int(scanner_state[2])
        chain_ids.append(chain_id)

    scanners_by_chain = {}
    for scanner, chain_id in zip(scanners, chain_ids):
        if chain_id not in scanners_by_chain:
            scanners_by_chain[chain_id] = []
        scanners_by_chain[chain_id].append(scanner)

    return bot_id, scanner, scanners_by_chain


def get_sharding_stats(chain_id):
    bot_id, scanner, scanners_by_chain = get_scanners()
    scanners = scanners_by_chain[chain_id] if chain_id in scanners_by_chain else []
    n_shards = len(scanners)
    shard_index = scanners.index(scanner) if scanner in scanners else 0
    return shard_index, n_shards


if __name__ == '__main__':
    print(get_sharding_stats(1))
