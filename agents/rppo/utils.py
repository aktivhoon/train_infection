import sys
import os
import gym
import gym_infection
def create_env(scenario:str):
    """Initializes an environment based on the provided scenario name.
    
    Args:
        scenario {str}: Name of the to be instantiated environment
    Returns:
        {env}: Returns the selected environment instance.
    """
    if scenario == 'dilemma':
        env = gym.make('InfectionDilemma-v0')
    elif scenario == 'dilemma2d':
        env = gym.make('InfectionDilemma-v1')
    elif scenario == 'simple':
        env = gym.make('InfectionSimple-v0')
    elif scenario == 'simple2d':
        env = gym.make('InfectionSimple-v1')
    elif scenario == 'simple-together':
        env = gym.make('InfectionSimpleTogether-v0')
    elif scenario == 'simple-together2d':
        env = gym.make('InfectionSimpleTogether-v1')
    else:
        raise NotImplementedError
    return env

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 
    Args:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training
    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)
    
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp