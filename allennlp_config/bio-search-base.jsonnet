{

   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),

    "dataset_reader": {
        "type": "bio_trigger",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": std.extVar("BERT_PRETRAINED_MODEL"),  # "bert-base-multilingual-cased",
                "use_starting_offsets": true,
                "do_lowercase": std.extVar("BERT_DO_LOWERCASE") == "true",  # false
                "max_pieces": std.extVar("MAX_PIECES"),  # 512
            }
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": std.extVar("BATCH_SIZE")  # 8
    },
    "model": {
        "type": "zsee",
        "balance": (std.extVar("BALANCE") == "true"),  # true
        "dropout": std.extVar("ENCODER_DROPOUT"),  # 0
        "embeddings_dropout": std.extVar("EMBEDDING_DROPOUT"),  # 0
        "encoder": {
            "type": "feedforward",
            "feedforward": {
                "input_dim": 768,
                "num_layers": std.extVar("NUM_LAYERS"),  # 1
                "hidden_dims": std.extVar("HIDDEN_DIMS"),  # 500
                "activations": {
                    "type": std.extVar("ACTIVATION")  # "relu"
                }
            }
        },
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained-only",
                    "pretrained_model": std.extVar("BERT_PRETRAINED_MODEL"),  #  "bert-base-multilingual-cased"
                    "requires_grad": (std.extVar("BERT_TRAIN") == "true"),  # false
                    "top_layer_only": (std.extVar("BERT_TOP_LAYER_ONLY") == "true")  # false
                }
            }
        }
    },
    "train_data_path": "/home/mahnerak/zsee/data/bio_tagged_triggers/English/BIO/training_with_neg_eg.txt",
    "validation_data_path": "/home/mahnerak/zsee/data/bio_tagged_triggers/English/BIO/dev_with_neg_eg.txt",
    "trainer": {
        "cuda_device": std.extVar("CUDA_DEVICE"),  # 0
        "grad_norm": std.extVar("GRAD_NORM"),  # 5.0
        "num_epochs": std.extVar("NUM_EPOCHS"),  # 200
        "optimizer": {
            "type": std.extVar("OPTIMIZER"),  # "dense_sparse_adam"
            "lr": std.extVar("LEARNING_RATE")  # 0.0001
        },
        "patience": 20,
        "validation_metric": "+token_level/averaged_f1"
    }
}