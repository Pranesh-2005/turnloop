def run_transform(payload):
    print("starting transform")
    result = {"stage": "transform", "size": len(payload)}
    print("finished transform")
    return result
