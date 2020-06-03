local alt(config, mode="noop") = (
    if mode == "noop"
        then config
        else {
            "type": "alt",
            "mode": mode,
            "token_indexer": config
        }
);

{
    get(H, alt_mode="noop"):: {
        "pretrained_transformer": alt({
            "type": "pretrained_transformer_mismatched",
            "model_name": H.pretrained_model,
            "max_length": H.max_pieces
        }, mode=alt_mode)
    }
}
