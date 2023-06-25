import ray

@ray.remote
class VirtualClientEngineActor:

    def __init__(self, actor_id: int):
        
        self.actor_id = actor_id
    
    def run(self, client_fn, client_id):
        return client_id, client_fn()
