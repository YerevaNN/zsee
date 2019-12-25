# General options
local verbose = false;
local balance = true;
local batch_size = 10;
local learning_rate = std.extVar("LEARNING_RATE");  # default: 0.0001

# Multi-task setup
local alignment_loss_weight = std.extVar("ALIGNMENT_LOSS_WEIGHT");  # default: 1
local parallel_hops = 1;
local triplet_loss_margin = std.extVar("TRIPLET_LOSS_MARGIN");  # default: 100

# Embeddings
local truncate_long_sequences = true;
local top_layer_only = false;  # std.extVar("BERT_LAYERS") == "top"

# Data
local dataset_version = "with_neg_eg";  # "wout_neg_eg";  # std.extVar("DATASET_VERSION"); # "with_neg_eg" or "wout_neg_eg"
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

# Dataset Readers
local bio_trigger_reader = {
    "type": "bio_trigger",
    "token_indexers": token_indexers
};
local moses_parallel_reader = {
    "type": "moses_parallel",
    "token_indexers": token_indexers,
    "tokenizer": {
        "type": "word",
        "word_splitter": {
            "type": "corenlp_remote"
        }
    }
};
local default_reader = bio_trigger_reader;

# Data Iterators
local bio_trigger_iterator = {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": batch_size
};
local moses_parallel_iterator = {
    "type": "bucket",
    "sorting_keys": [["source_snt", "num_tokens"],
                     ["target_snt", "num_tokens"]],
    "batch_size": 2 * batch_size
};
local default_iterator = bio_trigger_iterator;


local data = {
    "datasets": {
        "en_train": {
            "path": "data/bio_tagged_triggers/English/BIO/training_%s.txt" % dataset_version
        },
        "validation": {
            "path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % dataset_version
        },
        "en_test": {
            "path": "data/bio_tagged_triggers/English/BIO/test_%s.txt" % dataset_version
        },

        "ar_train": {
            "path": "data/bio_tagged_triggers/Arabic/BIO/train_%s.txt" % dataset_version
        },
        "ar_validation": {
            "path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % dataset_version
        },
        "ar_test": {
            "path": "data/bio_tagged_triggers/Arabic/BIO/test_%s.txt" % dataset_version
        },

        "parallel_train": {
            "path": "data/News-Commentary.train.ar-en.ar",
            "dataset_reader": moses_parallel_reader,
            "data_iterator": moses_parallel_iterator
        },
        "parallel_validation": {
            "path": "data/News-Commentary.valid.ar-en.ar",
             "dataset_reader": moses_parallel_reader,
             "data_iterator": moses_parallel_iterator
        }
    },
    "datasets_for_training": ["en_train", "parallel_train"],
    "default_dataset_reader": bio_trigger_reader,
    "default_data_iterator": bio_trigger_iterator,
    "data_mixer": {
        "type": "round-robin",
        "hops": {
            "en_train": 1,
            "parallel_train": parallel_hops
        },
        "until_any_finished": true
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
local enforce_orthogonality = 0;  # default: 0.01
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
local num_encoder_layers = std.parseInt(std.extVar("NUM_ENCODER_LAYERS"));  # default: 1
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

    "trainer": {
        "type": "callback",

        "num_epochs": 200,
        "shuffle": true,
        "cuda_device": 0,
        "optimizer": {
            "type": "bert_adam",
            "lr": learning_rate,

# Primarily for BERT whole model fine-tuning
#           "t_total": 1000,
#           "schedule": "warmup_constant",
#           "max_grad_norm": 1.0,
#           "weight_decay": 0.01,
#           "parameter_groups": [
#               [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
#           ],
        },
        "callbacks": [
            {
                "type": "validate",
                "on": "validation"
            },
            {
                "type": "validate",
                "on": "ar_validation"
            },
            {
                "type": "validate",
                "on": "parallel_validation"
            },
            {
                "type": "checkpoint",
                "checkpointer": {
                    "num_serialized_models_to_keep": 1
                }
            },
            {
                "type": "track_metrics",
                "patience": 100,
                "validation_metric": "+seqeval_f1_score"
            },
            {
                "type": "log_to_tensorboard",
                "log_batch_size_period": 10
            }
        ] + extra_callbacks
    }
}