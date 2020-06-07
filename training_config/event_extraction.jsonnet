local libs = {
    cache: import 'libs/cache.libsonnet',
    data: import 'libs/data.libsonnet',
    model: import 'libs/model.libsonnet',
    util: import 'libs/util.libsonnet',
    hparams: import 'libs/hparams.libsonnet'
};
local L = libs;

local hparams = L.hparams.get();
local H = hparams;

local readers = {
    "bio_trigger_reader": L.data.bio_trigger_reader(H) + {
        "filter_instances": H.filter_train_instances
    },
    "moses_parallel_reader": L.data.moses_parallel_reader(H),

    # Readers with integrated translators
    # Temporary only
    "bio_trigger_reader_en-de": L.data.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="de"),
    "bio_trigger_reader_en-ar": L.data.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="ar"),
    "bio_trigger_reader_en-ru": L.data.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="ru"),
    "bio_trigger_reader_en-hy": L.data.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="hy"),
    "bio_trigger_reader_ar-en": L.data.bio_trigger_reader_with_translation(H, source_lang="ar", target_lang="en"),
    "bio_trigger_reader_ar-de": L.data.bio_trigger_reader_with_translation(H, source_lang="ar", target_lang="de"),

    # Alts

    "bio_trigger_reader_alt": L.data.bio_trigger_reader(H, vocab_alt="all") + {
        "filter_instances": H.filter_train_instances
    },
    "moses_parallel_reader_alt": L.data.moses_parallel_reader(H, vocab_alt="all"),

    "bio_trigger_reader_en-de_alt": L.data.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="de", vocab_alt="all"),
    "bio_trigger_reader_en-ar_alt": L.data.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="ar", vocab_alt="all"),
    "bio_trigger_reader_en-ru_alt": L.data.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="ru", vocab_alt="all"),
    "bio_trigger_reader_en-hy_alt": L.data.bio_trigger_reader_with_translation(H, source_lang="en", target_lang="hy", vocab_alt="all"),
    "bio_trigger_reader_ar-en_alt": L.data.bio_trigger_reader_with_translation(H, source_lang="ar", target_lang="en", vocab_alt="all"),
    "bio_trigger_reader_ar-de_alt": L.data.bio_trigger_reader_with_translation(H, source_lang="ar", target_lang="de", vocab_alt="all"),
};

local validation_readers = readers + {
    "bio_trigger_reader": L.data.bio_trigger_reader(H),
    "moses_parallel_reader": L.data.moses_parallel_reader(H)  # TODO "instances_per_epoch": 500
};





# TODO merge concat and switch maybe? almost sure!
local dataset_reader = {
    "type": "concat",
    "dataset_reader": {
        "type": "switch",
        "readers": readers
    }
};

local validation_dataset_reader = {
    "type": "concat",
    "dataset_reader": {
        "type": "switch",
        "readers": validation_readers,
        "default": "bio_trigger_reader"
    }
};


local training_data_loader = {
    "batch_sampler": {
        "type": "homogeneous",
        "batch_sampler": {
            "type": if H.bucketing then "bucket" else "basic",
            "sampler": "random"
        },
        "batch_size": H.batch_size,
        "task_switcher": {
            // "type": "uniform_sampler"
            "type": "multihop",
            "hops": {
                "en_train": 1,
                "en-de_parallel_train": 0
            },
            // "max_epoch_iterations": 2000
        },
        "num_loops_per_epoch": H.loops_per_epoch
    }
};

local validation_data_loader = {
    "batch_sampler": {
        "type": "homogeneous",
        "batch_sampler": {
            "type": if H.validation_bucketing then "bucket" else "basic",
            "sampler": "sequential"
        },
        "batch_size": H.validation_batch_size,
        "task_switcher": "chain"
    },
};

 # [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],

local validation_metric = if H.validation_metric != null then H.validation_metric
                          else "+./" + H.validation_dataset + "/averaged_f1";

local is_large_model = std.length(std.findSubstr('-large', H.pretrained_model) + 
                                  std.findSubstr('1280', H.pretrained_model) + 
                                  std.findSubstr('mrm8488', H.pretrained_model)) > 0;
# mrm8488

{
    "random_seed": H.seed * 100 + 70,
    "numpy_seed": H.seed * 10 + 7,
    "pytorch_seed": H.seed,

    "hparams": H,


    "dataset_reader": dataset_reader,
    "validation_dataset_reader": validation_dataset_reader,


    "model": {
        "type": "multi-task",
        "loss_weights": [H.primary_loss_weight, H.alignment_loss_weight],
        "models": [
            L.model.event_extraction(H),
            L.model.alignment(H)
        ]
    },

    "trainer": {
        "num_epochs": H.num_epochs * H.loops_per_epoch,
        "cuda_device": 0,

        "optimizer":  {
            "type": H.optimizer,
            "lr": H.learning_rate,
            "weight_decay": H.weight_decay,
            "parameter_groups": [
                
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
                  @'\._pooler\.',
                  @'\._mapper\.',
                  @'\._projection\.',], {}],
            ],
        },


        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },

        "tensorboard_writer": {
            "summary_interval": 300,
            "should_log_learning_rate": true
        },


        "patience": 100,
        "validation_metric": validation_metric,

        "learning_rate_scheduler": if H.STR then {
            "type": "slanted_triangular",
            "gradual_unfreezing": H.gradual_unfreezing,
            "discriminative_fine_tuning": H.discriminative_fine_tuning
        } else null  # TODO better
    },

    "data_loader": training_data_loader,
    "validation_data_loader": validation_data_loader, # TODO biggest_batch_first


    "train_data_path": L.util.stringify([
        {
            "name": "en_train",
            "reader": "bio_trigger_reader"
                      + (if H.translate_train then "_en-ar" else "")
                      + (if H.train_alt then "_alt" else ""), # _en-ar
            "file_path": "data/bio_tagged_triggers/English/BIO/training_%s.txt" % H.dataset_version,
        }
        // {
        //     "name": "en-de_parallel_train",
        //     "reader": "moses_parallel_reader",
        //     "file_path": "data/News-Commentary.valid.de-en.de"
        // },
    ]),

    "validation_data_path": L.util.stringify([
        {
            "name": "en_val",
            "reader": "bio_trigger_reader",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version
        },
        {
            "name": "ar_val",
            "reader": "bio_trigger_reader",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version
        },
        {
            "name": "en_val_de",
            "reader": "bio_trigger_reader_en-de",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "en_val_ar",
            "reader": "bio_trigger_reader_en-ar",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "en_val_ru",
            "reader": "bio_trigger_reader_en-ru",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "en_val_hy",
            "reader": "bio_trigger_reader_en-hy",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "ar_val_en",
            "reader": "bio_trigger_reader_ar-en",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "ar_val_de",
            "reader": "bio_trigger_reader_ar-de",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "en_test",
            "reader": "bio_trigger_reader",
            "file_path": "data/bio_tagged_triggers/English/BIO/test_%s.txt" % H.dataset_version
        },
        {
            "name": "ar_test",
            "reader": "bio_trigger_reader",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/test_%s.txt" % H.dataset_version
        },
        {
            "name": "en_test_ar",
            "reader": "bio_trigger_reader_en-ar",
            "file_path": "data/bio_tagged_triggers/English/BIO/test_%s.txt" % H.dataset_version
        },
        {
            "name": "ar_test_en",
            "reader": "bio_trigger_reader_ar-en",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/test_%s.txt" % H.dataset_version
        },
    ] + if H.train_alt then [
        {
            "name": "en_val_alt",
            "reader": "bio_trigger_reader_alt",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version
        },
        {
            "name": "ar_val_alt",
            "reader": "bio_trigger_reader_alt",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version
        },
        {
            "name": "en_val_de_alt",
            "reader": "bio_trigger_reader_en-de_alt",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "en_val_ar_alt",
            "reader": "bio_trigger_reader_en-ar_alt",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "en_val_ru_alt",
            "reader": "bio_trigger_reader_en-ru_alt",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "en_val_hy_alt",
            "reader": "bio_trigger_reader_en-hy_alt",
            "file_path": "data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "ar_val_en_alt",
            "reader": "bio_trigger_reader_ar-en_alt",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "ar_val_de_alt",
            "reader": "bio_trigger_reader_ar-de_alt",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/dev_%s.txt" % H.dataset_version,
        },
        {
            "name": "en_test_alt",
            "reader": "bio_trigger_reader_alt",
            "file_path": "data/bio_tagged_triggers/English/BIO/test_%s.txt" % H.dataset_version
        },
        {
            "name": "ar_test_alt",
            "reader": "bio_trigger_reader_alt",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/test_%s.txt" % H.dataset_version
        },
        {
            "name": "en_test_ar_alt",
            "reader": "bio_trigger_reader_en-ar_alt",
            "file_path": "data/bio_tagged_triggers/English/BIO/test_%s.txt" % H.dataset_version
        },
        {
            "name": "ar_test_en_alt",
            "reader": "bio_trigger_reader_ar-en_alt",
            "file_path": "data/bio_tagged_triggers/Arabic/BIO/test_%s.txt" % H.dataset_version
        },
    ] else [])
    
    
    #"data/bio_tagged_triggers/English/BIO/dev_%s.txt" % H.dataset_version,
}