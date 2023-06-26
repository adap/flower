import ray

@ray.remote
class VirtualClientEngineActor:

    def __init__(self, actor_id: int):
        
        self.actor_id = actor_id
    
    def run(self, client_fn, client_id):
        # execute tasks and return result
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        return client_id, client_fn()
