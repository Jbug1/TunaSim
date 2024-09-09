
eval_db = nist20
generate_entropy_intensity_stats = False
buckets_per_item = 100
if generate_entropy_intensity_stats:

    eval_db = pd.read_pickle(eval_db)
    eval_db['entropy'] = [shannon_ent(i[:,1]) for i in eval_db['spectrum']]

    bucket_thresh_mz = np.linspace(min(eval_db['precursor']), max(eval_db['precursor']), buckets_per_item)
    bucket_array_entropy = np.zeros(buckets_per_item)
    items_seen_entropy = np.zeros(buckets_per_item)

    for i in range(len(eval_db)):

        position = bisect_left(bucket_thresh_mz,eval_db.iloc[i]['precursor'])
        bucket_array_entropy[position] += eval_db.iloc[i]['entropy']
        items_seen_entropy[position] += 1

    bucket_array_entropy /= items_seen_entropy
    plt.plot(bucket_thresh_mz,bucket_array_entropy)
    plt.title('With Precursor')
    plt.ylabel('Entropy')
    plt.xlabel('m/z')
    plt.show()

    bucket_array_entropy = np.zeros(buckets_per_item)
    items_seen_entropy = np.zeros(buckets_per_item)

    clean_specs = list()
    for i in range(len(eval_db)):
        clean_specs.append(tools.clean_spectrum(eval_db.iloc[i]['spectrum'], 
                                                 max_mz = eval_db.iloc[i]['precursor']-tools.ppm(eval_db.iloc[i]['precursor'],10)))
        
    for i in range(len(eval_db)):

        position = bisect_left(bucket_thresh_mz,eval_db.iloc[i]['precursor'])
        bucket_array_entropy[position] += shannon_ent(clean_specs[i][:,1])
        items_seen_entropy[position] += 1

    bucket_array_entropy /= items_seen_entropy
    plt.plot(bucket_thresh_mz,bucket_array_entropy)
    plt.title('Without Precursor')
    plt.ylabel('Entropy')
    plt.xlabel('m/z')
    plt.show()


    # for j in [0,10,20,40]:

    #     bucket_array_entropy = np.zeros(buckets_per_item)
    #     items_seen_entropy = np.zeros(buckets_per_item)

    #     eval_db_ = eval_db[eval_db['collision_energy']==j]

    #     for i in range(len(eval_db_)):

    #         position = bisect_left(bucket_thresh_mz,eval_db_.iloc[i]['precursor'])
    #         bucket_array_entropy[position] += eval_db_.iloc[i]['entropy']
    #         items_seen_entropy[position] += 1

    #     bucket_array_entropy /= items_seen_entropy
    #     plt.plot(bucket_thresh_mz,bucket_array_entropy)
    #     plt.title(f'With Precursor CE:{j}')
    #     plt.ylabel('Entropy')
    #     plt.xlabel('m/z')
    #     plt.show()


    buckets_per_item = 10000
    bucket_thresh_mz = np.linspace(min(eval_db['precursor']), max(eval_db['precursor']), buckets_per_item)
    bucket_array_entropy = np.zeros(buckets_per_item)
    items_seen_entropy = np.zeros(buckets_per_item)
    for i in clean_specs:

        for j in i:

            position = bisect_left(bucket_thresh_mz,j[0])
            bucket_array_entropy[position] += j[1]
            items_seen_entropy[position] += 1

    bucket_array_entropy /= items_seen_entropy
    plt.plot(bucket_thresh_mz,bucket_array_entropy)
    plt.title('Intensity by Bucket')
    plt.ylabel('Mean Intensity')
    plt.xlabel('m/z')
    plt.show()

    buckets_per_item = 10000
    bucket_thresh_mz = np.linspace(min(eval_db['precursor']), max(eval_db['precursor']), buckets_per_item)
    bucket_array_entropy = np.zeros(buckets_per_item)
    items_seen_entropy = np.zeros(buckets_per_item)
    for i in clean_specs:

        i = tools.clean_spectrum(i, standardize=True)

        for j in i:

            position = bisect_left(bucket_thresh_mz,j[0])
            bucket_array_entropy[position] += j[1]
            items_seen_entropy[position] += 1

    bucket_array_entropy /= items_seen_entropy
    plt.plot(bucket_thresh_mz,bucket_array_entropy)
    plt.title('Intensity by Bucket')
    plt.ylabel('Mean Intensity')
    plt.xlabel('m/z')
    plt.show()

    plt.plot()




    #Similarity methods and transformation parameters below. Leave sim methods as None to run all

chunk_size = 1e6
adduct_match = False
strong_self_separation = False

comparison_metrics = ['entropy',
                'manhattan',
                'lorentzian',
                'dot_product',
                'fidelity',
                'matusita',
                'chi2',
                'laplacian',
                'harmonic_mean',
                'bhattacharya_1',
                'squared_chord',
                'cross_ent']

ppm_windows = [10]
noise_threshes=[0.01,0.0]
centroid_tolerance_vals = [0.05,0.01]
centroid_tolerance_types=['da','da']
reweight_methods = ['orig',1]
reweight_names = ['orig',1]
sim_methods=comparison_metrics
prec_removes=[True]


fullRun=False
if fullRun:

    train_size=3e6
    test_size=1e6
    test_size=2e6

    max_matches=None
    adduct_match = False

    for i in ppm_windows:

        if build_dataset:

            #read in first bases and shuffle order
            query_train = pd.read_pickle(f'{outputs_path}/intermediateOutputs/splitMatches/first_query.pkl')
            query_train=query_train.sample(frac=1)

            target_=pd.read_pickle(target)
            if self_search:
                target_.insert(0,'queryID', [i for i in range(len(target_))])
            else:
                target_.insert(0,'queryID', ["*" for i in range(len(target_))])

            #restrict to only cores in query if both conditions are met
            if strong_self_separation and self_search:
                target_ = target_[np.isin(target_['inchi_base']), list(set(query_train['inchi_base']))]


            #create matches for model to train on
            os.mkdir(f'{outputs_path}/intermediateOutputs/splitMatches/train/{i}_ppm')
            datasetBuilder.create_matches_df_chunk(query_df = query_train,
                                                   target_df = target_,
                                                   precursor_thresh = i,
                                                   adduct_match = adduct_match,
                                                   max_len = train_size,
                                                   chunk_size= chunk_size,
                                                   outpath = f'{outputs_path}/intermediateOutputs/splitMatches/train/{i}_ppm',
                                                   logpath = f'{outputs_path}/intermediateOutputs/splitMatches/train/log_{i}_ppm.txt')
            
            del(query_train)
            del(target_)
            
            #need to implement
            os.mkdir(f'{outputs_path}/intermediateOutputs/modelDatasets/train/{i}_ppm')
            train = datasetBuilder.create_model_dataset_chunk(
                                                input_path = f'{outputs_path}/intermediateOutputs/splitMatches/train/{i}_ppm',
                                                output_path = f'{outputs_path}/intermediateOutputs/modelDatasets/train/{i}_ppm',
                                                sim_methods = sim_methods, 
                                                noise_threshes = noise_threshes, 
                                                centroid_tolerance_vals = centroid_tolerance_vals, 
                                                centroid_tolerance_types = centroid_tolerance_types,
                                                reweight_methods = reweight_methods,
                                                prec_removes = prec_removes
            )

            #read in first bases and shuffle order
            query_ = pd.read_pickle(f'{outputs_path}/intermediateOutputs/splitMatches/second_query.pkl')
            query_ = query_train.sample(frac=1)

            target_=pd.read_pickle(target)
            if self_search:
                target_.insert(0,'queryID', [i for i in range(len(target_))])
            else:
                target_.insert(0,'queryID', ["*" for i in range(len(target_))])

            #restrict to only cores in query if both conditions are met
            if strong_self_separation and self_search:
                target_ = target_[np.isin(target_['inchi_base']), list(set(query_train['inchi_base']))]


            #create matches for model to train on
            os.mkdir(f'{outputs_path}/intermediateOutputs/splitMatches/train/{i}_ppm')
            datasetBuilder.create_matches_df_chunk(query_df = query_train,
                                                   target_df = target_,
                                                   precursor_thresh = i,
                                                   adduct_match = adduct_match,
                                                   max_len = train_size,
                                                   chunk_size= chunk_size,
                                                   outpath = f'{outputs_path}/intermediateOutputs/splitMatches/train/{i}_ppm',
                                                   logpath = f'{outputs_path}/intermediateOutputs/splitMatches/train/log_{i}_ppm.txt')
            
            del(query_train)
            del(target_)
            
            #need to implement
            os.mkdir(f'{outputs_path}/intermediateOutputs/modelDatasets/train/{i}_ppm')
            train = datasetBuilder.create_model_dataset_chunk(
                                                input_path = f'{outputs_path}/intermediateOutputs/splitMatches/train/{i}_ppm',
                                                output_path = f'{outputs_path}/intermediateOutputs/modelDatasets/train/{i}_ppm',
                                                sim_methods = sim_methods, 
                                                noise_threshes = noise_threshes, 
                                                centroid_tolerance_vals = centroid_tolerance_vals, 
                                                centroid_tolerance_types = centroid_tolerance_types,
                                                reweight_methods = reweight_methods,
                                                prec_removes = prec_removes
            )
            #read in first bases and shuffle order
            query_test = pd.read_pickle(f'{outputs_path}/intermediateOutputs/splitMatches/third_query.pkl')
            query_test=query_test.sample(frac=1)

            #create matches for model to train on
            matches = datasetBuilder.create_matches_df(query_test,target_,i,max_matches,test_size, adduct_match)
            matches.to_pickle(f'{outputs_path}/intermediateOutputs/splitMatches/test_matches_{i}_ppm.pkl')
            del(query_test)
            
            test = datasetBuilder.create_cleaned_df(
                                                matches, 
                                                sim_methods, 
                                                noise_threshes, 
                                                centroid_tolerance_vals, 
                                                centroid_tolerance_types,
                                                powers,
                                                prec_removes
            )

            test.to_pickle(f'{outputs_path}/intermediateOutputs/splitMatches/test_cleaned_matches_{i}_ppm.pkl')
            del(test)
       




fullRun = True
generate_unnormed = True
if fullRun:
    for i in ppm_windows:

        train_matches = pd.read_pickle(f'{outputs_path}/intermediateOutputs/splitMatches/train_cleaned_matches_{i}_ppm.pkl')

        testUtils.create_model_and_ind_data(train_matches,
                                            f'{outputs_path}/intermediateOutputs/modelDatasets/train/{i}_ppm',
                                            comparison_metrics,
                                            unnormed = False)
        
        if generate_unnormed:
            testUtils.create_model_and_ind_data(train_matches,
                                            f'{outputs_path}/intermediateOutputs/modelDatasets/train/unnormed_{i}_ppm',
                                            comparison_metrics,
                                            unnormed = True)
    
        del(train_matches)
        print('created train data')

        val_matches = pd.read_pickle(f'{outputs_path}/intermediateOutputs/splitMatches/val_cleaned_matches_{i}_ppm.pkl')

        testUtils.create_model_and_ind_data(val_matches,
                                            f'{outputs_path}/intermediateOutputs/modelDatasets/val/{i}_ppm',
                                            comparison_metrics,
                                            unnormed = False)
        
        if generate_unnormed:
            testUtils.create_model_and_ind_data(val_matches,
                                            f'{outputs_path}/intermediateOutputs/modelDatasets/val/unnormed_{i}_ppm',
                                            comparison_metrics,
                                            unnormed = True)
    
        del(val_matches)
        print('created validation data')

        test_matches = pd.read_pickle(f'{outputs_path}/intermediateOutputs/splitMatches/test_cleaned_matches_{i}_ppm.pkl')

        testUtils.create_model_and_ind_data(test_matches,
                                            f'{outputs_path}/intermediateOutputs/modelDatasets/test/{i}_ppm',
                                            comparison_metrics,
                                            unnormed = False)
        
        if generate_unnormed:
            testUtils.create_model_and_ind_data(test_matches,
                                            f'{outputs_path}/intermediateOutputs/modelDatasets/tes/unnormed_{i}_ppm',
                                            comparison_metrics,
                                            unnormed = True)
    
        del(test_matches)
        print('created test data')

        