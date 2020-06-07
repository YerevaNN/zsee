local libs = {
    optimizer: import 'optimizer.libsonnet'
};

local L = libs;

{
    get(H, validation_metric=null):: {

        "num_epochs": H.num_epochs * H.loops_per_epoch,
        "cuda_device": 0,

        "optimizer":  {
            "type": H.optimizer,
            "lr": H.learning_rate,
            "weight_decay": H.weight_decay,
            "parameter_groups": L.optimizer.get(H),
        },

        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },

        "tensorboard_writer": {
            "summary_interval": 300,
            "should_log_learning_rate": true
        },

        "patience": H.patience,
        "validation_metric": validation_metric,

        "learning_rate_scheduler": if H.STR then {
            "type": "slanted_triangular",
            "gradual_unfreezing": H.gradual_unfreezing,
            "discriminative_fine_tuning": H.discriminative_fine_tuning
        } else null  # TODO better
    },
}