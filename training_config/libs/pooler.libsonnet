local default_error = error "no default was provided";

{
    get(H, name)::
        local available_poolers = {
            'mean': {
                "type": "bag_of_embeddings",
                "averaged": true
            },
            'attention': {
                "type": "attention",
                "projection": false
            },
            'bert_cls': {
                "type": "bert_pooler",
                "pretrained_model": H.pretrained_model
            },
            'just_cls': {
                "type": "cls_pooler",
            },
        };
        local pooler_name = H[name];
        

        if pooler_name != null
            then available_poolers[pooler_name]
            else null
}
