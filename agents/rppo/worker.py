import multiprocessing
import multiprocessing.connection
from utils import create_env
import numpy as np

def worker_process(remote: multiprocessing.connection.Connection, env_name:str) -> None:
    """Executes the threaded interface to the environment.
    
    Args:
        remote {multiprocessing.connection.Connection} -- Parent thread
        env_name {str} -- Name of the to be instantiated environment
    """
    # Spawn environment
    try:
        env = create_env(env_name)
    except KeyboardInterrupt:
        pass
    total_reward = np.array([0] * len(env.world.agents))
    # Communication interface of the environment thread
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                total_reward = np.array(reward) + total_reward
                remote.send((obs,reward,done,info))
            elif cmd == "reset":
                remote.send((env.reset(), total_reward))
                total_reward = np.array([0] * len(env.world.agents))
            elif cmd == "close":
                remote.send(env.close())
                remote.close()
                break
            else:
                raise NotImplementedError
        except:
            break

class Worker:
    """A worker that runs one environment on one thread."""
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process
    
    def __init__(self, env_name:str):
        """
        Args:
            env_name (str) -- Name of the to be instantiated environment
        """
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_name))
        self.process.start()