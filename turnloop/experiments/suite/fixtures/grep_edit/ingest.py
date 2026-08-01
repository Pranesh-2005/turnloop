def run_ingest(payload):
    print("starting ingest")
    result = {"stage": "ingest", "size": len(payload)}
    print("finished ingest")
    return result
