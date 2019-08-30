local get_int_var(x) = std.parseInt(std.extVar(x));

local glove_embedding_dim = 300;
local use_glove_embedding = get_int_var('use_glove_embedding'); # 1; # extVar

local char_embedding_dim = get_int_var('char_embedding_dim'); # default: 100; # extVar
local char_embedding_ngram_filter_sizes = [5, 4, 3];
local use_char_embedding = get_int_var('use_char_embedding'); # default: 1; # extVar

local elmo_embedding_dim = 1024;
local use_elmo_embedding = get_int_var('use_elmo_embedding'); # default: 0; # extVar

local bert_embedding_dim = 768;
local use_bert_embedding = get_int_var('use_bert_embedding'); # default: 1; # extVar

local spacy_model = std.extVar('spacy_model'); # default: "en_core_sci_sm"; # extVar

local encoding_dim = get_int_var('encoding_dim'); # default: 400; // embedding_dim; // 100;
local num_encoder_layers = get_int_var('num_encoder_layers'); # default: 2; # extVar
local encoder_inner_dropout = get_int_var('encoder_inner_dropout') / 100; # default: 0.0; # extVar
local encoder_outer_dropout = get_int_var('encoder_outer_dropout') / 100; # default: 0.0; # extVar

local embeddings_dropout = get_int_var('embeddings_dropout') / 100; # default: 0.0; # extVar

local batch_size = get_int_var('batch_size'); # default: 16; # extVar

local grad_norm = if get_int_var('grad_norm') > 0 then get_int_var('grad_norm') else null; # default: 5.0; # extVar

local optimizer = std.extVar('optimizer'); # default: "adam";
local lr = get_int_var('lr') / 10000; # default: 0.001; # extVar
local balance = if get_int_var('balance') > 0 then true else false;

local embedding_dim = use_glove_embedding   *   glove_embedding_dim
                    + use_char_embedding    *   char_embedding_dim  *   std.length(char_embedding_ngram_filter_sizes)
                    + use_elmo_embedding    *   elmo_embedding_dim
                    + use_bert_embedding    *   bert_embedding_dim;


{
    "dataset_reader": {
        "type": "ace2005_trigger",
        "token_indexers": (
            if use_glove_embedding == 0 then {} else {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": false
            }
        }) + (
            if use_char_embedding == 0 then {} else {
            "token_characters": {
                "type": "characters",
                "min_padding_length": 5
            }
        }) + (
            if use_elmo_embedding == 0 then {} else {
            "elmo": {
                "type": "elmo_characters"
            }
        }) + (
            if use_bert_embedding == 0 then {} else {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": 'bert-base-multilingual-cased',
                "do_lowercase": false,
                "use_starting_offsets": true,
                "max_pieces": 4096
            }
        }),
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "spacy",
                "language": spacy_model
            }
        },
        "sentence_splitter": {
            "type": "spacy_raw"
        }
    },

    "train_data_path": 'data/LDC2006T06/en_train.files',
    "validation_data_path": 'data/LDC2006T06/en_dev.files',
    "test_data_path": 'data/LDC2006T06/en_test.files',

    "model": {
        "type": "zsee",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": (
                if use_glove_embedding == 0 then {} else {
                "tokens": ["tokens"]
            }) + (
                if use_char_embedding == 0 then {} else {
                "token_characters": ["token_characters"]
            }) + (
                if use_elmo_embedding == 0 then {} else {
                "elmo": ["elmo"]
            }) + (
                if use_bert_embedding == 0 then {} else {
                "bert": ["bert", "bert-offsets"]
            }),
            "token_embedders": (
                if use_glove_embedding == 0 then {} else {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
                    "embedding_dim": glove_embedding_dim,
                    "trainable": false
                }
            }) + (
                if use_char_embedding == 0 then {} else {
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "num_embeddings": 262,
                        "embedding_dim": 16
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 16,
                        "num_filters": char_embedding_dim,
                        "ngram_filter_sizes": char_embedding_ngram_filter_sizes
                    }
                }
            }) + (
                if use_elmo_embedding == 0 then {} else {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false
                }
            }) + (
                if use_bert_embedding == 0 then {} else {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base-multilingual-cased",
                    "requires_grad": true,
                    "top_layer_only": false
                }
            })
        },
        //    "encoder": {
        //        "type": "pass_through",
        //        "input_dim": embedding_dim
        //    },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": encoding_dim,
            "num_layers": num_encoder_layers,
            "bidirectional": true,
            "dropout": encoder_inner_dropout
        },
        //    "encoder": {
        //        "type": "stacked_bidirectional_lstm",
        //        "input_size": embedding_dim,
        //        "hidden_size": encoding_dim,
        //        "num_layers": 1,
        //        "layer_dropout_probability": 0.4,
        //        "recurrent_dropout_probability": 0.4
        //    },
        "embeddings_dropout": embeddings_dropout,
        "dropout": encoder_outer_dropout,
        "balance": balance
    },
    "iterator": {
        "type": "basic",
        //    "sorting_keys": [["text", "num_tokens"]],
        //    "padding_noise": 0.0,
        "batch_size": batch_size,
        //    "biggest_batch_first": true
    },
    "trainer": {
        "num_epochs": 50,
        "grad_norm": grad_norm,
        "patience" : 20,
        "cuda_device" : 0,
        "validation_metric": "+averaged_f1",
        //    "learning_rate_scheduler": {
        //      "type": "reduce_on_plateau",
        //      "factor": 0.5,
        //      "mode": "max",
        //      "patience": 2
        //    },
        "optimizer": {
            "type": optimizer,
            "lr": lr
        },
        "num_serialized_models_to_keep": 1
    }
}