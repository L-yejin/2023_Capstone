def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert'):
        args.mode = 'train' # 'train','test
        args.test_model_path = None # test path

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 128
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        
        # Data Augmentation parameter 지정 부분 → 아직 추가해야됨. args만 만들어둔 거
        args.data_type = 'origin_dataset' # 'origin_dataset','noise_dataset','similarity','redundancy'
        args.N_Aug = None # [5, 10, 15]
        args.P = None # [None, 0.1, 0.2, 0.3]

        # data type이 noise인 경우, 아래 항목 중 선택
        args.type_noise_item = 'all_item' # ['all_item','popular_item']
        args.type_noise_item_size = 300
        
        # Embedding 방법 선택
        args.model_embedding = 'origin_embedding' # 'hyper_embedding'

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda' # 'cpu'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert'
        args.model_init_seed = 0

        args.bert_dropout = 0.1 # 0.5
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 4

