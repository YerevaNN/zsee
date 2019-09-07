local glove_embedding_dim = 300;
local char_embedding_dim = 100; // 0; // 100;
local elmo_embedding_dim = 0; // 1024;
local bert_embedding_dim  = 768; //0; //768;

local embedding_dim = glove_embedding_dim + 3 * char_embedding_dim + elmo_embedding_dim + bert_embedding_dim ;
local encoding_dim = 400; // embedding_dim; // 100;

{
  "dataset_reader": {
    "type": "ace2005_trigger",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 5
      },
//      "elmo": {
//        "type": "elmo_characters"
//      },
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": 'bert-base-multilingual-cased',
          "do_lowercase": false,
          "use_starting_offsets": true,
          "max_pieces": 4096
      }
    },
    "tokenizer": {
        "type": "word",
        "word_splitter": {
            "type": "spacy",
//            "language": "en_core_sci_sm"
        }
    },
    "sentence_splitter": {
        "type": "spacy_raw"
    }
//    "max_instances_to_read": 10
  },

  "train_data_path": 'data/LDC2006T06/en_train.files',
  "validation_data_path": 'data/LDC2006T06/en_dev.files',
  "test_data_path": 'data/LDC2006T06/en_test.files',

  "model": {
    "type": "zsee",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
        "token_characters": ["token_characters"],
        "tokens": ["tokens"],
        "elmo": ["elmo"]
      },
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
            "embedding_dim": glove_embedding_dim,
            "trainable": false
        },
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
                "ngram_filter_sizes": [5, 4, 3]
            }
        },
//        "elmo":{
//            "type": "elmo_token_embedder",
//            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
//            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
//            "do_layer_norm": false,
////            "dropout": 0.5
//        },
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": 'bert-base-multilingual-cased'
        }
      }
    },
//    "encoder": {
//        "type": "pass_through",
//        "input_dim": embedding_dim
//    },
    "encoder": {
        "type": "lstm",
        "input_size": embedding_dim,
        "hidden_size": encoding_dim,
        "num_layers": 2,
        "bidirectional": true,
        "dropout": 0.0
    },
//    "encoder": {
//        "type": "stacked_bidirectional_lstm",
//        "input_size": embedding_dim,
//        "hidden_size": encoding_dim,
//        "num_layers": 1,
//        "layer_dropout_probability": 0.4,
//        "recurrent_dropout_probability": 0.4
//    },
    "embeddings_dropout": 0.0,
    "dropout": 0.0,
    "balance": true

  },
  "iterator": {
    "type": "basic",
//    "sorting_keys": [["text", "num_tokens"]],
//    "padding_noise": 0.0,
    "batch_size": 16,
//    "biggest_batch_first": true
  },
  "trainer": {
    "num_epochs": 1000,
//    "grad_norm": 5.0,
    "patience" : 300,
    "cuda_device" : 0,
    "validation_metric": "+averaged_f1",
//    "learning_rate_scheduler": {
//      "type": "reduce_on_plateau",
//      "factor": 0.5,
//      "mode": "max",
//      "patience": 2
//    },
    "optimizer": {
      "type": "adam"
    },
    "num_serialized_models_to_keep": 4
  }
}