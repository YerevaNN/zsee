local lib = {
    pooler: import 'pooler.libsonnet'
};

local mapper(H) = {
    "num_layers": H.num_mapper_layers,
    "hidden_dims": H.mapper_dim,
    "activations": if H.num_mapper_layers > 0
                       then std.makeArray(H.num_mapper_layers - 1, function(i) H.mapper_activation) + [H.mapper_last_activation]
                       else [],
};

{
    event_extraction(H):: {
        "type": "sentence_level_zsee",

        "balance": H.balance,
        "verbose": H.verbose,
        "dropout": 0,
        "embeddings_dropout": 0,
        "encoder": "pass_through",
        "softmax": true,
        "beta": H.beta,
        "gamma": H.gamma,
        "projection": true,
        "initializer": {},

        "pooler": lib.pooler.get(H, 'pooler'),

        "text_field_embedder": {
            "type": "mapped",
            "text_field_embedder": {
                "token_embedders": {
                    "pretrained_transformer": {
                        "type": if H.mismatched_embedder then "pretrained_transformer_mismatched" else "pretrained_transformer",
                        "model_name": H.pretrained_model,
#                            "requires_grad": H.finetune_bert,  # TODO
#                            "top_layer_only": H.top_layer_only,  # TODO
                        "max_length": H.max_pieces
                    },
                }
            },
            "mapper": mapper(H),
            "pre_normalization": H.mapping_pre_normalization,
            "post_normalization": H.mapping_post_normalization
        }
    },



    alignment(H):: {
        "type": "embeddings_alignment",
        "map_both": true,
        "distance": "mse",
        "pooler": lib.pooler.get(H, 'alignment_pooler'),
        "triplet_loss_margin": H.triplet_loss_margin,
        "initializer": {
#  "regexes": [] + extra_initializers # TODO
        },
        "verbose": H.verbose,
        "version": H.version,
        "symmetric": H.alignment_symmetric
    }
}
