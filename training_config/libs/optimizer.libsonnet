{
    get(H):: 
        local is_large_model = std.length(std.findSubstr('-large', H.pretrained_model) + 
                               std.findSubstr('1280', H.pretrained_model) + 
                               std.findSubstr('mrm8488', H.pretrained_model)) > 0;
        [
                
        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 
        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 

        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 
        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 

        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 
        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 

        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 
        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 

        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 
        [[],{}], [[],{}], [[],{}], [[],{}], [[],{}], 

        [[@'\.transformer_model\.embeddings.',
            @'\.transformer_model\.lang_embeddings\.',
            @'\.transformer_model\.position_embeddings\.',
            @'\.transformer_model\.layer_norm_emb\.'
            ], {}],

        [[@'\.(attentions|layer|ffns|layer_norm\d)\.0\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.1\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.2\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.3\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.4\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.5\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.6\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.7\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.8\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.9\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.10\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.11\.'], {}],
    ] + (if is_large_model then [
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.12\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.13\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.14\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.15\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.16\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.17\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.18\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.19\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.20\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.21\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.22\.'], {}],
        [[@'\.(attentions|layer|ffns|layer_norm\d)\.23\.'], {}],
    ] else []) + [
        [[@'\.pooler\.',
            @'s\._pooler\.',
            @'\._mapper\.',
            @'\._projection\.',], {}],
    ]
}

