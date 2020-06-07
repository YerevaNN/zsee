{
    training(H):: {
        "batch_sampler": {
            "type": "homogeneous",
            "batch_sampler": {
                "type": if H.bucketing then "bucket" else "basic",
                "sampler": "random"
            },
            "batch_size": H.batch_size,
            "task_switcher": H.task_switcher, # TODO
            // TODO
            // {
            //     // "type": "uniform_sampler"
            //     "type": "multihop",
            //     "hops": {
            //         "en_train": 1,
            //         "en-de_parallel_train": 0
            //     },
            //     // "max_epoch_iterations": 2000
            // },
            "num_loops_per_epoch": H.loops_per_epoch
        }
    },

    validation(H)::{
        "batch_sampler": {
            "type": "homogeneous",
            "batch_sampler": {
                "type": if H.validation_bucketing then "bucket" else "basic",
                "sampler": "sequential"
            },
            "batch_size": H.validation_batch_size,
            "task_switcher": "chain"
        },
    }
}