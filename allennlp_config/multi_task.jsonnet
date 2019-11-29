# General options
local verbose = false;
local balance = true;
local batch_size = 20;
local learning_rate = 0.0001;

# Multi-task setup
local alignment_loss_weight = 0;
local hops = 1;
local triplet_loss_margin = 1000000;

# Embeddings
local truncate_long_sequences = false;
local top_layer_only = false;  # (std.extVar("BERT_LAYERS") == "top"); # "top" or "mix"

# Data
local dataset_version = "wout_neg_eg";  # std.extVar("DATASET_VERSION"); # "with_neg_eg" or "wout_neg_eg"
local token_indexers = {
    "bert": {
       "type": "bert-pretrained",
       "do_lowercase": false,
       "max_pieces": 512,
       "pretrained_model": "bert-base-multilingual-cased",
       "use_starting_offsets": true,
       "truncate_long_sequences": truncate_long_sequences
    }
};
local moses_parallel_reader = {
    "type": "moses_parallel",
    "token_indexers": token_indexers,
    "tokenizer": {
        "type": "word",
        "word_splitter": {
            "type": "corenlp_remote"
        }
    },
    "limit_samples": 100000,
    //                "source_field": "text",
    //                "target_field": "translation"
    //        "lazy": true
};
local data = {
    "paths": {
        "en_train": "data/bio_tagged_triggers/English/BIO/training_" + dataset_version + ".txt",
        "validation": "data/bio_tagged_triggers/English/BIO/dev_" + dataset_version + ".txt",
        "en_test": "data/bio_tagged_triggers/English/BIO/test_" + dataset_version + ".txt",

        "ar_train": "data/bio_tagged_triggers/Arabic/train.txt",
        "ar_validation": "data/bio_tagged_triggers/Arabic/dev.txt",
        "ar_test": "data/bio_tagged_triggers/Arabic/test.txt",

        "zh_train": "data/bio_tagged_triggers/Chinese/train.txt",
        "zh_validation": "data/bio_tagged_triggers/Chinese/dev.txt",
        "zh_test": "data/bio_tagged_triggers/Chinese/test.txt",

        "parallel_train": "data/News-Commentary.train.ar-en.ar",
        "parallel_validation": "data/News-Commentary.valid.ar-en.ar"
    },
    "datasets_for_training": ["en_train", "parallel_train"],
    //        "datasets_for_vocab_creation": ["trigger_train"],
    "mingler": {
        "type": "round-robin"
    },
    "dataset_reader": {
        "type": "bio_trigger",
        "token_indexers": token_indexers
    },
    "dataset_readers": {
        # More specific dataset readers
        "parallel_train": moses_parallel_reader,
        "parallel_validation": moses_parallel_reader
    },
    "iterator": {
        "type": "homogeneous_batch",
        "batch_size": batch_size,
        "partition_key": "dataset",
        "hops": {
            "en_train": 1,
            "parallel_train": hops
        },
        "until_finished": "any"
        },
        //        "validation_iterator": {
        //            "type": "bucket",
        //            "sorting_keys": [["text", "num_tokens"]],
        //            "padding_noise": 0,
        //            "batch_size": 30,
        //        },
        "iterators": {
            # More specific data iterators
        }
    };

# Mapper
local num_mapper_layers = 0;
local mapper_dim = 100;
local mapping_dim = if num_mapper_layers == 0 then 768 else mapper_dim;
local mapper_activation = "relu";
local mapper = (
    if num_mapper_layers == 0 then
        null
    else {
        "input_dim": 768,
        "num_layers": num_mapper_layers,
        "hidden_dims": mapping_dim,
        "activations": std.makeArray(num_mapper_layers - 1, function(i) mapper_activation) + ["linear"]
    }
);

# Mapper constraints
local enforce_orthogonality = 0; # 0.01
local extra_callbacks = (if enforce_orthogonality > 0 then [{
    "type": "orthonormalize",
    "regex": "_mapper._linear_layers.0.weight",
    "beta": enforce_orthogonality
}] else []);
local extra_initializers = (if enforce_orthogonality > 0 then [
    ["_mapper._linear_layers.0.weight", "eye"],
    ["_mapper._linear_layers.0.bias", "zero"]
] else []);

# Encoder
local num_encoder_layers = 1;  # std.parseInt(std.extVar("NUM_LAYERS")); # 0, 1, ...
local encoder_dim = 1000;
local encoder = (
    if num_encoder_layers == 0 then {
        "type": "pass_through",
        "input_dim": mapping_dim
    } else {
        "type": "feedforward",
        "feedforward": {
            "input_dim": mapping_dim,
            "num_layers": num_encoder_layers,
            "hidden_dims": encoder_dim,
            "activations": {
                "type": "relu"
            }
        }
    }
);


{
    "data": data,
    "model": {
        "type": "multi-task",
        "loss_weights": [1, alignment_loss_weight],
        "models": [{
            "type": "zsee",
            "balance": balance,
            "verbose": verbose,
            "dropout": 0,
            "embeddings_dropout": 0,
            "encoder": encoder,

            "text_field_embedder": {
                "type": "mapped",
//                "frozen_embeddings": "false",  # pretrained bert scalar mix is trainable
                "text_field_embedder": {
                    "allow_unmatched_keys": true,
                    "embedder_to_indexer_map": {
                        "bert": ["bert", "bert-offsets"]
                    },
                    "token_embedders": {
                        "bert": {
                            "type": "bert-pretrained-only",
                            "pretrained_model": "bert-base-multilingual-cased",
                            "requires_grad": false,
                            "top_layer_only": top_layer_only
                        }
                    }
                },
                "mapper": mapper
            }
        }, {
            "type": "embeddings_alignment",
            "map_both": "true",
            "pooler": {
                "type": "bag_of_embeddings",
                "embedding_dim": mapping_dim,
                "averaged": true
            },
            "distance": {
                "type": "mse"
            },
            "triplet_loss_margin": triplet_loss_margin,
            "initializer": [] + extra_initializers
        }]
    },
 #  "data/bio_tagged_triggers/English/BIO/training_" + dataset_version + ".txt",
//    "validation_data_path": "data/bio_tagged_triggers/English/BIO/dev_" + dataset_version + ".txt",

//    "other_data_paths": {
////        "chinese": "data/bio_tagged_triggers/Chinese/dev.txt",
//        "arabic": "data/bio_tagged_triggers/Arabic/dev.txt"
//    },
//    "datasets_for_vocab_creation": ["train"],



    "trainer": {
        "type": "multi-task",

        "num_epochs": 600,
        "shuffle": true,
        "cuda_device": 0,
        "optimizer": {
            "type": "bert_adam",
//            "type": "bert_adam",
            "lr": learning_rate,   # <--
//            "t_total": 1000,
//            "schedule": "warmup_constant",
//            "max_grad_norm": 1.0,
//            "weight_decay": 0.01,
//            "parameter_groups": [
//              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
//            ],
        },
        "callbacks": [{
            "type": "validate",
            "on": "validation"
        }, {
            "type": "validate",
            "on": "zh_validation"
        }, {
            "type": "validate",
            "on": "ar_validation"
        }, {
            "type": "validate",
            "on": "parallel_validation"
        }, {
            "type": "checkpoint",
            "checkpointer": {
                "num_serialized_models_to_keep": 1000
            }
        }, {
            "type": "track_metrics",
            "patience": 100,
            "validation_metric": "+seqeval_f1_score"
        }, {
            "type": "log_to_tensorboard",
            "log_batch_size_period": 10
        }] + extra_callbacks
    }
}