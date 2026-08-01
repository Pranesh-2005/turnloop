def run_load(payload):
    print("starting load")
    result = {"stage": "load", "size": len(payload)}
    print("finished load")
    return result
