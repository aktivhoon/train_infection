def simple2d_config():
    return {
        "env": "simple2d",
        "gamma": 0.999,
        "lamda": 0.95,
        "updates": 10000,
        "reward_scale":0.1,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 8,
        "value_loss_coefficient": 0.25,
        "hidden_layer_size": 32,
        "recurrence": 
            {
            "sequence_length": 1,
            "hidden_state_size": 16,
            "layer_type": "lstm",
            "reset_hidden_state": False
            },
        "learning_rate_schedule":
            {
            "initial": 2.0e-4,
            "final": 2.0e-4,
            "power": 1.0,
            "max_decay_steps": 300
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.001,
            "power": 1.0,
            "max_decay_steps": 300
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 300
            }
    }