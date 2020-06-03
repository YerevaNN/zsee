local defaults = {
    'seed': 133,

    # General options
    'verbose': true,

    # Fight data imbalance
    'balance': false,
    'pi': 0.01,
    'beta': 0,
    'gamma': 0,

    # Optimization
    'batch_size': 8,
    'validation_batch_size': 8,
    'learning_rate': 0.0001,

    'bucketing': false,
    'validation_bucketing': false,
    'num_epochs': 8,

    # Multi-task setup
    'primary_loss_weight': 1,
    'alignment_loss_weight': 0,
    #    'parallel_hops': 0,
    'triplet_loss_margin': 10,

    # Embeddings
    # Temporary disabled
    # 'truncate_long_sequences': true,
    # 'top_layer_only': false,

    # Data
    'dataset_version': "with_neg_eg",

    # Mapper
    'num_mapper_layers': 0,    # mapper is without output activation
    'mapper_dim': 768,
    'mapper_activation': 'relu',
    'mapper_last_activation': 'linear',

    # Optimizer
    'optimizer': 'huggingface_adamw',
    'STR': true,
    'weight_decay': 0, # TODO nonzero
    'gradual_unfreezing': false,
    'discriminative_fine_tuning': false,
    # TODO grad norm   

    // 'pretrained_model_dim': 768,
    'finetune': true,  # TODO implement

    'pretrained_model': "bert-base-multilingual-cased",
    'mismatched_embedder': true,

    'pooler': 'just_cls',
    'alignment_pooler': null,
    'alignment_symmetric': false,

    'filter_train_instances': false,
    'max_pieces': 256,

    'mapping_pre_normalization': null,
    'mapping_post_normalization': null,

    'loops_per_epoch': 1,
    'version': 2,

    'parallel_data': 'en-de_parallel_train',

    'train_alt': false,
    'translate_train': false,
    'validation_metric': null,
    'validation_dataset': 'en_val'
};

{
    defaults: defaults,
    extras: std.parseJson(std.extVar('HPARAMS')),
    
    get()::
        local default_hparams = self.defaults;
        local extras = self.extras;

        local hparams = default_hparams + extras;
        # Make sure that no extra arguments are passed
        assert (std.length(hparams) == std.length(default_hparams));

        hparams,
}