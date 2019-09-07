{
    "dataset_reader": {
        "type": "ace2005_trigger",
        "sentence_splitter": {
            "type": "spacy_raw"
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "do_lowercase": false,
                "max_pieces": 4096,
                "pretrained_model": "bert-base-multilingual-cased",
                "use_starting_offsets": true
            },
//            "tokens": {
//                "type": "single_id",
//                "lowercase_tokens": false
//            }
        },
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "spacy",
                "language": "en_core_web_sm"
            }
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8
    },
    "model": {
        "type": "zsee",
        "balance": true,
        "dropout": 0,
        "embeddings_dropout": 0,
        "encoder": {
//            "type": "lstm",
//            "bidirectional": true,
//            "dropout": 0,
//            "hidden_size": 400,
//            "input_size": 768,
//            "num_layers": 2
            "type": "pass_through",
            "input_dim": 768
        },
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": [
                    "bert",
                    "bert-offsets"
                ],
//                "tokens": [
//                    "tokens"
//                ]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base-multilingual-cased",
                    "requires_grad": false,
                    "top_layer_only": false
                },
//                "tokens": {
//                    "type": "embedding",
//                    "embedding_dim": 300,
//                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
//                    "trainable": false
//                }
            }
        }
    },
    "train_data_path": "data/LDC2006T06/en_train.files",
    "validation_data_path": "data/LDC2006T06/en_dev.files",
    "trainer": {
        "cuda_device": 0,
        "grad_norm": null,
        "num_epochs": 50,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "dense_sparse_adam",
            "lr": 0.0003
        },
        "patience": 20,
        "validation_metric": "+averaged_f1"
    }
}