local lib = {
    cache: import 'cache.libsonnet',
    token_indexers: import 'token_indexers.libsonnet'
};

local L = lib;

local readers = {
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
    }, enabled=cache),
};

{
    readers: readers,

    training_readers(H):: {
        "bio_trigger_reader": readers.bio_trigger_reader(H) + {
            "filter_instances": H.filter_train_instances
        },
        "moses_parallel_reader": readers.moses_parallel_reader(H),

        # Readers with integrated translators
        # Temporary only
        "bio_trigger_reader_en-de": readers.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="de"),
        "bio_trigger_reader_en-ar": readers.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="ar"),
        "bio_trigger_reader_en-ru": readers.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="ru"),
        "bio_trigger_reader_en-hy": readers.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="hy"),
        "bio_trigger_reader_ar-en": readers.bio_trigger_reader_with_translation(H, source_lang="ar", target_lang="en"),
        "bio_trigger_reader_ar-de": readers.bio_trigger_reader_with_translation(H, source_lang="ar", target_lang="de"),

        # Alts

        "bio_trigger_reader_alt": readers.bio_trigger_reader(H, vocab_alt="all") + {
            "filter_instances": H.filter_train_instances
        },
        "moses_parallel_reader_alt": readers.moses_parallel_reader(H, vocab_alt="all"),

        "bio_trigger_reader_en-de_alt": readers.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="de", vocab_alt="all"),
        "bio_trigger_reader_en-ar_alt": readers.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="ar", vocab_alt="all"),
        "bio_trigger_reader_en-ru_alt": readers.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="ru", vocab_alt="all"),
        "bio_trigger_reader_en-hy_alt": readers.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="hy", vocab_alt="all"),
        "bio_trigger_reader_ar-en_alt": readers.bio_trigger_reader_with_translation(H, source_lang="ar", target_lang="en", vocab_alt="all"),
        "bio_trigger_reader_ar-de_alt": readers.bio_trigger_reader_with_translation(H, source_lang="ar", target_lang="de", vocab_alt="all"),
    },


    validation_readers(H)::
        self.training_readers(H) + {
            "bio_trigger_reader": readers.bio_trigger_reader(H),
            "moses_parallel_reader": readers.moses_parallel_reader(H)  # TODO "instances_per_epoch": 500
        },

    get_training_dataset_reader(H)::
        local training_readers = self.training_readers(H);
        {
            "type": "concat",
            "dataset_reader": {
                "type": "switch",
                "readers": training_readers
            }
        },

    get_validation_dataset_reader(H, default_reader):: 
        local validation_readers = self.validation_readers(H);
        {
            "type": "concat",
            "dataset_reader": {
                "type": "switch",
                "readers": validation_readers,
                "default": default_reader
            }
        },
}
