# Cache config
local manifest(config) = std.manifestPython(config);
local hash(config, salt) = std.md5(salt + manifest(config));
local cache_config(config, salt='axuhac', cache_dir='caches') = config + {cache_directory: (cache_dir + '/' + hash(config, salt))};

local default_hparams = {
    'seed': 133,

    # General options
    'verbose': true,

    # Fight data imbalance
    'balance': false,
    'pi': 0.01,
    'beta': 0,
    'gamma': 0,

    # Optimization
    'batch_size': 15,
    'validation_batch_size': self.batch_size,
    'learning_rate': 0.0001,

    'bucketing': false,
    'validation_bucketing': true,
    'num_epochs': 30,

    # Multi-task setup
    'primary_loss_weight': 1,
    'alignment_loss_weight': 0,
    'parallel_hops': 0,
    'triplet_loss_margin': 10,

    # Embeddings
    'truncate_long_sequences': true,
    'top_layer_only': false,

    # Data
    'dataset_version': "with_neg_eg",

    # Mapper
    'num_mapper_layers': 1,
    'mapper_dim': 768,

    # Optimizer
    'optimizer': 'adam',
    # In case of if AdamW is enabled
    'STR': false,
    'weight_decay': 0,

    'finetune_bert': false,
    'bert_model': "bert-base-multilingual-cased",
    'bert_do_lowercase': false,
    'pooler': 'attention',
    'alignment_pooler': 'mean',
    'alignment_symmetric': false,

    'filter_train_instances': false,
    'max_pieces': 256,

    'mapping_pre_normalization': null,
    'mapping_post_normalization': null,

    'miniepochs': false,
    'version': 2,

    'parallel_data': 'en-de_parallel_train'
};
local extras = std.parseJson(std.extVar('HPARAMS'));

local hparams = default_hparams + extras;
assert (std.length(hparams) == std.length(default_hparams));
local H = hparams;


local token_indexers = {
    "bert": {
        "type": "bert-pretrained",
        "do_lowercase": H.bert_do_lowercase,
        "max_pieces": H.max_pieces,
        "pretrained_model": H.bert_model,
        "use_starting_offsets": true,
        "truncate_long_sequences": H.truncate_long_sequences
    }
};

# Dataset Readers
local bio_trigger_reader = {
    "type": "bio_trigger",
    "token_indexers": token_indexers,
    "null_label": true,
    "show_progress": true,
    "sentence_level_only": true
};

local bio_trigger_reader_with_translation(source_lang, target_lang) = {
    "type": "bio_trigger",
    "token_indexers": token_indexers,
    "null_label": true,
    "show_progress": true,
    "translation_service": {
        "type": "cached",
        "cache_dir": "data/mt_2/",
        "tokenizer": "corenlp",
        "source_lang": source_lang,
        "target_lang": target_lang
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
    }
};

local default_reader = bio_trigger_reader;

# Data Iterators
# TODO bucketing for training only

local training_iterator = {
    "type": if H.bucketing then "bucket" else "basic",
    "batch_size": H.batch_size
};
local validation_iterator = if H.validation_bucketing then {
    "type": "bucket",
    "batch_size": H.validation_batch_size,
    "padding_noise": 0,
    "biggest_batch_first": true
} else {
    "type": "basic",
    "batch_size": H.batch_size
};
local default_iterator = training_iterator; # Just in case


local data = {
    "datasets": {
        "en_train": {
            "path": "data/bio_tagged_triggers/English/BIO/training_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader + {
                "filter_instances": H.filter_train_instances
            }),
            "data_iterator": training_iterator + (if H.miniepochs then {
                "instances_per_epoch": 2000
            } else {})
        },
        "validation": {
            "path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version
        },
        "en_test": {
            "path": "data/bio_tagged_triggers/English/BIO/test_%s.txt" % H.dataset_version
        },

        "ar_train": {
            "path": "data/bio_tagged_triggers/Arabic/BIO/train_%s.txt" % H.dataset_version
        },
        "ar_validation": {
            "path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version
        },
        "ar_test": {
            "path": "data/bio_tagged_triggers/Arabic/BIO/test_%s.txt" % H.dataset_version
        },

        #  translated versions
        "en_train_de": {
            "path": "data/bio_tagged_triggers/English/BIO/training_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader_with_translation("en", "de")),
            "data_iterator": validation_iterator + {
                "instances_per_epoch": 500
            }
        },
        "en_train_ar": {
            "path": "data/bio_tagged_triggers/English/BIO/training_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader_with_translation("en", "ar"))
        },

        "en_validation_de": {
            "path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader_with_translation("en", "de"))
        },
        "en_validation_ar": {
            "path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader_with_translation("en", "ar"))
        },
        "en_validation_ru": {
            "path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader_with_translation("en", "ru"))
        },
        "en_validation_hy": {
            "path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader_with_translation("en", "hy"))
        },


        "ar_validation_en": {
            "path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader_with_translation("ar", "en"))
        },
        "ar_validation_de": {
            "path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version,
            "dataset_reader": cache_config(bio_trigger_reader_with_translation("ar", "de"))
        },

        "en-ar_parallel_train": {
            "path": "data/News-Commentary.train.ar-en.ar",
            "dataset_reader": cache_config(moses_parallel_reader),
            "data_iterator": training_iterator
        },
        "en-ar_parallel_validation": {
            "path": "data/News-Commentary.valid.ar-en.ar",
            "dataset_reader": cache_config(moses_parallel_reader),
            "data_iterator": validation_iterator + {
                "instances_per_epoch": 500
            }
        },

        "en-de_parallel_train": {
            "path": "data/News-Commentary.train.de-en.de",
            "dataset_reader": cache_config(moses_parallel_reader),
            "data_iterator": training_iterator
        },
        "en-de_parallel_validation": {
            "path": "data/News-Commentary.valid.de-en.de",
            "dataset_reader": cache_config(moses_parallel_reader),
            "data_iterator": validation_iterator + {
                "instances_per_epoch": 500
            }
        },
    },
    "datasets_for_training": ["en_train", H.parallel_data],
    "datasets_for_vocab_creation": ["en_train"],
    "default_dataset_reader": cache_config(bio_trigger_reader),
    "default_data_iterator": validation_iterator,
    "data_mixer": {
        "type": "round-robin",
        "hops": {
            "en_train": 1,
            [H.parallel_data]: H.parallel_hops
        },
        "until_any_finished": true
    }
};

local mapping_dim = if H.num_mapper_layers == 0 then 768 else H.mapper_dim;
local mapper_activation = "relu";
local mapper = (
    if H.num_mapper_layers == 0 then
        null
    else {
        "input_dim": 768,
        "num_layers": H.num_mapper_layers,
        "hidden_dims": mapping_dim,
        "activations": std.makeArray(H.num_mapper_layers - 1, function(i) mapper_activation) + ["linear"]
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
//local num_encoder_layers = std.parseInt(std.extVar("NUM_ENCODER_LAYERS"));  # default: 1
//local encoder_dim = 768;
//local encoder = {
//    "input_dim": mapping_dim,
//    "num_layers": num_encoder_layers,
//    "hidden_dims": encoder_dim,
//    "activations": {
//        "type": "relu"
//    }
//};

local available_poolers = {
    'mean': {
        "type": "bag_of_embeddings",
        "embedding_dim": mapping_dim,
        "averaged": true
    },
    'attention': {
        "type": "attention",
        "input_dim": mapping_dim,
        "projection": false
    },
    'gru': {
        "type": "gru",
        "bidirectional": true,
        "hidden_size": mapping_dim / 2,
        "input_size": mapping_dim,  // elmo (1024) + cnn (100) + num_context_answers (2) * marker_embedding_dim (10)
        "num_layers": 1
    },
    'max': {
        # Not implemented yet
    },
    'bert_cls': {
        "type": "bert_pooler",
        "pretrained_model": H.bert_model,
        "requires_grad": true
    },
    'just_cls': {
        "type": "first_token",
        "embedding_dim": mapping_dim
    },
};

local downstream_pooler = available_poolers[H.pooler];
local alignment_pooler = (if H.alignment_pooler == null then null else (
                              if H.alignment_pooler == 'copy'
                              then downstream_pooler
                              else available_poolers[H.alignment_pooler]));

local scheduler_callbacks = if H.STR then [{
    "type": "update_learning_rate",
    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": H.num_epochs,
        "num_steps_per_epoch": std.ceil(21000 / H.batch_size),  # TODO
    }
}] else [];


{

    "random_seed": H.seed * 100 + 70,
    "numpy_seed": H.seed * 10 + 7,
    "pytorch_seed": H.seed,

    "hparams": H,
    "data": data,
    "model": {
        "type": "multi-task",
        "loss_weights": [H.primary_loss_weight, H.alignment_loss_weight],
        "models": [{
            "type": "sentence_level_zsee",
            "balance": H.balance,
            "verbose": H.verbose,
            "dropout": 0,
            "embeddings_dropout": 0,
            "encoder": {
                "type": "pass_through",
                "input_dim": mapping_dim
            },
            "softmax": true,
            "beta": H.beta,
            // "gamma": H.gamma,
            "projection": true,
            "initializer": [],
            //    ["_projection.bias", {
            //        "type": "constant",
            //        "val": -std.log((1-pi)/pi)
            //    }]

            "pooler": downstream_pooler,

            "text_field_embedder": {
                "type": "mapped",
                "text_field_embedder": {
                    "token_embedders": {
                        "bert": {
                            "type": "frozen-bert",  # "bert-pretrained-only"
                            "pretrained_model": H.bert_model,
                            "requires_grad": H.finetune_bert,
                            "top_layer_only": H.top_layer_only,
                            "max_pieces": H.max_pieces
                        },
                    }
                },
                "mapper": mapper,
                "pre_normalization": H.mapping_pre_normalization,
                "post_normalization": H.mapping_post_normalization
            }
        }, {
            "type": "embeddings_alignment",
            "map_both": "true",
            "distance": {
                "type": "mse"
            },
            "pooler": alignment_pooler,
            "triplet_loss_margin": H.triplet_loss_margin,
            "initializer": [] + extra_initializers,
            "verbose": H.verbose,
            "version": H.version,
            "symmetric": H.alignment_symmetric
        }]
    },

    "trainer": {
        "type": "callback",

        "num_epochs": H.num_epochs + 5,
        "shuffle": true,
        "cuda_device": 0,

        "optimizer": (if H.optimizer == "adamw" then {
            "type": "huggingface_adamw",
            "lr": H.learning_rate,
            "correct_bias": false,
            "weight_decay": H.weight_decay,
            "parameter_groups": [
                [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        } else {
            "type": H.optimizer,
            "lr": H.learning_rate
        }),
        "callbacks": [
            {
                "type": "validate",
                "on": "validation"
            },
            {
                "type": "validate",
                "on": "ar_validation"
            },

            # Validate on translations

//            {
//                "type": "validate",
//                "on": "en_train"
//            },



            {
                "type": "validate",
                "on": "en_train_de"
            },

            {
                "type": "validate",
                "on": "en_train_ar"
            },



            {
                "type": "validate",
                "on": "en_validation_de"
            },
//
            {
                "type": "validate",
                "on": "en_validation_ar"
            },
            {
                "type": "validate",
                "on": "en_validation_ru"
            },
            {
                "type": "validate",
                "on": "en_validation_hy"
            },
//
            {
                "type": "validate",
                "on": "ar_validation_en"
            },

//            {
//                "type": "validate",
//                "on": "ar_validation_de"
//            },



            {
                "type": "validate",
                "on": "en-de_parallel_validation"
            },
            {
                "type": "validate",
                "on": "en-ar_parallel_validation"
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
                "validation_metric": "+averaged_f1"
            },
            {
                "type": "log_to_tensorboard",
                "log_batch_size_period": 10,
                "summary_interval": 300,
                "should_log_learning_rate": true
            },
            {
                "type": "log_singular_values",
                "regex": "_mapper._linear_layers.0.weight"
            }
        ] + extra_callbacks + scheduler_callbacks
    }
}