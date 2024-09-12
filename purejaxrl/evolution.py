import os

os.environ['MUJOCO_GL'] = 'egl'

import jax
import jax.numpy as jnp
from flax import linen as nn
from evosax import CMA_ES, ParameterReshaper, FitnessShaper
from evosax.utils import ESLog
import wandb
import functools
import evosax 

import numpy as np

from train_for_evolution import vmap_train_for_evolution


if __name__ == "__main__":
    
    wandb.init(
        #config=FLAGS,
        entity='naumix',
        project='Evolution_new',
        group=f'HalfCheetah',
        name=f'New_Test')
    
    max_steps = 10000000
    popsize = 8
    num_generations = 100
    
    param_reshaper = ParameterReshaper(jnp.zeros(6))
    strategy = CMA_ES(popsize=popsize, num_dims=param_reshaper.total_params, elite_ratio=0.5)
    es_params = strategy.default_params
    es_logging = ESLog(param_reshaper.total_params, num_generations, top_k=5, maximize=True)
    log = es_logging.initialize()
    fit_shaper = FitnessShaper(centered_rank=False, z_score=True, w_decay=0.0, maximize=True)

    rng = jax.random.PRNGKey(0)
    rng, rng_init = jax.random.split(rng)
    state = strategy.initialize(rng_init, es_params)
    
    log_data = np.zeros((popsize*num_generations, 6+1))
    
    @jax.jit
    def transform_x(x):
        x = nn.tanh(x)
        return x
    
    @jax.jit 
    def evolve(rng: jax.random.PRNGKey, state: evosax.strategies.cma_es.EvoState, log: dict, seed: int):
        
        config1 = {
            "LR": 3e-4,
            "NUM_ENVS": 512,
            "NUM_STEPS": 64,
            "TOTAL_TIMESTEPS": max_steps,
            "UPDATE_EPOCHS": 3,
            "NUM_MINIBATCHES": 32,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.0,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "halfcheetah",
            "ANNEAL_LR": False,
            "NORMALIZE_ENV": True,
            "DEBUG": False,
        }
        
        config2 = {
            "LR": 3e-4,
            "NUM_ENVS": 512,
            "NUM_STEPS": 64,
            "TOTAL_TIMESTEPS": max_steps,
            "UPDATE_EPOCHS": 3,
            "NUM_MINIBATCHES": 32,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.0,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "ant",
            "ANNEAL_LR": False,
            "NORMALIZE_ENV": True,
            "DEBUG": False,
        }
        
        config3 = {
            "LR": 3e-4,
            "NUM_ENVS": 512,
            "NUM_STEPS": 64,
            "TOTAL_TIMESTEPS": max_steps,
            "UPDATE_EPOCHS": 3,
            "NUM_MINIBATCHES": 32,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.0,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "walker2d",
            "ANNEAL_LR": False,
            "NORMALIZE_ENV": True,
            "DEBUG": False,
        }
        
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_ask, state, es_params)
        evolved_params = transform_x(x)
        fitness1 = vmap_train_for_evolution(config1, seed, evolved_params)
        fitness2 = vmap_train_for_evolution(config2, seed, evolved_params)
        fitness3 = vmap_train_for_evolution(config3, seed, evolved_params)
        fitness = (fitness1 + fitness2 + fitness3) / 3
        #fitness = proj_mems.rewards.mean(-1).max(-1)     
        #proj_mems.rewards.shape
        fit_re = fit_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state, es_params)
        #log = es_logging.update(log, x, fitness)
        return rng, state, log, fitness, x

    for gen in range(num_generations):
        rng, state, log, fitness, x = evolve(rng, state, log, gen)
        log_data[popsize*gen:popsize*gen+popsize, 0:6] = np.array(x)
        log_data[popsize*gen:popsize*gen+popsize, 6] = np.array(fitness)
        np.savetxt("test.csv", log_data, delimiter=",")
        
        print(f"Generation: {gen}, Best: {log['log_top_1'][gen]}, Fitness: {fitness.mean()}")
        wandb.log({"Fitness": fitness.mean(),
                   "Best Fitness": fitness.max(),
                   "Best Value": x[jnp.argmax(fitness)],
                   "Variance": x.var()}, step=gen)
