local lib = {
    cache: import 'cache.libsonnet',
    token_indexers: import 'token_indexers.libsonnet'
};

local L = lib;


{
    bio_trigger_reader(H, vocab_alt="noop", cache=null):: L.cache.apply({
        "type": "bio_trigger",
        "token_indexers": L.token_indexers.get(H, alt_mode=vocab_alt),
        "null_label": true,
        "show_progress": true,
        "sentence_level_only": true
    }, enabled=cache),

    bio_trigger_reader_with_translation(H, source_lang, target_lang,
                                        vocab_alt="noop", cache=null):: L.cache.apply({
        "type": "bio_trigger",
        "token_indexers": L.token_indexers.get(H, alt_mode=vocab_alt),
        "null_label": true,
        "show_progress": true,
        
        "translation_service": {
            "type": "cached",
            "cache_dir": "data/mt_2/",
            "tokenizer": "corenlp",
            "source_lang": source_lang,
            "target_lang": target_lang
        }
    }, enabled=cache),

    moses_parallel_reader(H, vocab_alt="noop", cache=null):: L.cache.apply({
        "type": "moses_parallel",
        "token_indexers": L.token_indexers.get(H, alt_mode=vocab_alt),
        "tokenizer": "corenlp"
    }, enabled=cache)
}
