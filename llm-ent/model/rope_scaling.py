def apply_rope_scaling(config, factor=2.0, rope_type="linear"):
    config.rope_scaling = {"type": rope_type, "factor": float(factor)}
    return config
