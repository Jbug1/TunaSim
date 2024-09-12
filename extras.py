
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


def create_model_dataset(
    matches_df,
    sim_methods=None,
    noise_threshes=[0.01],
    centroid_tolerance_vals=[0.05],
    centroid_tolerance_types=["da"],
    reweight_methods=[None],
    prec_removes=[None],
    prec_remove_names = ['none'],
    original_order=False
):
    """ """
    # create helper vars
    out_df = None
    spec_columns = [
        "ent_query",
        "npeaks_query",
        "normalent_query",
        "ent_target",
        "npeaks_target",
        "normalent_target",
    ]

    # create initial value spec columns
    for remove in range(len(prec_removes)):

        init_spec_df = matches_df.apply(
            lambda x: get_spec_features(
                x["query"], x["target"]
            ),
            axis=1,
            result_type="expand",
        )

        init_spec_df.columns = spec_columns

        ticker = 0
        for i in noise_threshes:
            for j in reweight_methods:
                for k in range(len(centroid_tolerance_vals)):

                    ticker += 1
                    if ticker % 10 == 0:
                        print(f"added {ticker} settings")

                    spec_columns_ = [
                        f"{x}_n:{i} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{j}, pr:{prec_remove_names[remove]}"
                        for x in spec_columns
                    ]


                    sim_columns_ = [
                         f"{x}_n:{i} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{j}, pr:{prec_remove_names[remove]}"
                        for x in sim_methods
                    ]

                    # clean specs and get corresponding spec features
                    cleaned_df = matches_df.apply(
                        lambda x: clean_and_spec_features(
                            x["query"],
                            x["precquery"],
                            x["target"],
                            x["prectarget"],
                            noise_thresh=i,
                            centroid_thresh=centroid_tolerance_vals[k],
                            centroid_type=centroid_tolerance_types[k],
                            reweight_method=j,
                            prec_remove=prec_removes[remove],
                            original_order=original_order
                        ),
                        axis=1,
                        result_type="expand",
                    )


                    cleaned_df.columns = (
                        spec_columns_  + ["query", "target"]
                    )

                    
                    # create columns of similarity scores
                    if centroid_tolerance_types[k] == "ppm":
                        sim_df = cleaned_df.apply(
                            lambda x: get_sim_features(
                                x["query"],
                                x["target"],
                                sim_methods,
                                ms2_ppm=centroid_tolerance_vals[k],
                                original_order=original_order,
                                reweight_method=j
                            ),
                            axis=1,
                            result_type="expand",
                        )

                        sim_df.columns = sim_columns_

                    else:
                        
                        sim_df = cleaned_df.apply(
                            lambda x: get_sim_features(
                                x["query"],
                                x["target"],
                                sim_methods,
                                ms2_da=centroid_tolerance_vals[k],
                                original_order=original_order,
                                reweight_method=j
                            ),
                            axis=1,
                            result_type="expand",
                        )
                        
                        sim_df.columns = sim_columns_

                    # add everything to the output df
                    if out_df is None:

                        out_df = pd.concat(
                            (
                                matches_df.iloc[:, :-3],
                                init_spec_df,
                                cleaned_df.iloc[:, :-2],
                                sim_df,
                            ),
                            axis=1,
                        )

                    else:

                        out_df = pd.concat(
                            (
                                out_df,
                                cleaned_df.iloc[:, :-2],
                                sim_df,
                            ),
                            axis=1,
                        )

    out_df["match"] = matches_df["match"]
    return out_df

def clean_and_spec_features_single(
    spec1,
    prec1,
    noise_thresh,
    centroid_thresh,
    centroid_type="ppm",
    reweight_method=1,
    verbose=False
):
    """
    Function to clean the query and target specs according to parameters passed. Returns only matched spec
    """

    if verbose:
        print(spec1)
    
    if centroid_type == "ppm":

        spec1_ = tools.clean_spectrum(
            spec1,
            noise_removal=noise_thresh,
            ms2_ppm=centroid_thresh,
            standardize=False,
            max_mz=prec1,
        )

    else:
        spec1_ = tools.clean_spectrum(
            spec1, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=False
        )

    if verbose:
        print(spec1_)
        

    # reweight by given reweight_method
    spec1_[:,1]= tools.weight_intensity(spec1_[:,1], reweight_method)
    # print(spec1_)

    # get new spec features
    spec_features = get_spec_features_single(spec1_, prec1)
    if verbose:
        print(spec_features)

    spec1_ = tools.standardize_spectrum(spec1_)

    return spec_features + [spec1_]


def get_spec_features_single(spec, precursor):

    if len(spec) == 0:
        spec = np.array([[1, 0]])

    outrow = np.zeros(4)

    # first get all peaks below precursor mz
    below_prec_indices = np.where(spec[:, 0] < (precursor - tools.ppm(precursor, 3)))
    mass_reduction = np.sum(spec[below_prec_indices][:, 1]) / np.sum(spec[:, 1])

    spec = spec[below_prec_indices]

    n_peaks = len(spec)
    ent = scipy.stats.entropy(spec[:, 1])

    outrow[0] = ent
    outrow[1] = n_peaks

    if n_peaks < 2:
        outrow[2] = -1
    else:
        outrow[2] = ent / np.log(n_peaks)
    outrow[3] = mass_reduction

    return list(outrow)

def get_sim_features_all(targets, queries, sim_methods, ms2_ppm=None, ms2_da=None):
    """
    This function calculates the similarities of the queries (one parameter setting) against all target specs
    """

    if ms2_da is None and ms2_ppm is None:
        raise ValueError("need either ms2da or ms2ppm to proceed")

    sims_out = None

    for i in range(targets.shape[1]):

        temp = pd.concat((targets.iloc[:, i : i + 1], queries), axis=1)

        col0 = temp.columns[0]
        col1 = temp.columns[1]

        sims = temp.apply(
            lambda x: get_sim_features(
                x[col0], x[col1], methods=sim_methods, ms2_da=ms2_da, ms2_ppm=ms2_ppm
            ), 
            axis=1,
            result_type="expand"
        )

        if sims_out is None:
            sims_out = sims
        else:
            sims_out = pd.concat((sims_out, sims), axis=1)

    return sims_out

def create_matches_df(query_df, target_df, precursor_thresh, max_len, adduct_match, logpath=None):

    non_spec_columns = [
        "precquery",
        "prectarget",
        "cequery",
        "cetarget",
        "instsame",
        "ceratio",
        "ceabs",
        "prec_abs_dif",
        "prec_ppm_dif",
        "mass_reduction_query",
        "mass_reduction_target",
        "mass_reduc_abs",
        "mass_reduc_ratio"
    ]

    start = time.perf_counter()
   
    #to be sure...shuffle query
    #query_df = query_df.sample(frac=1)

    out = None
    #target_df = target_df.sample(frac=1)
    printy = 1e5

    target_df['spectrum'] = [np.array(i, dtype=np.float64) for i in target_df['spectrum']]
    query_df['spectrum'] = [np.array(i, dtype=np.float64) for i in query_df['spectrum']]

    seen=0
    seen_=0
    unmatched=0
    cores_set=set()
    query_num=list()
    for i in range(len(query_df)):

        seen_+=1
        
        cores_set.add(query_df.iloc[i]['inchi_base'])

        if adduct_match:
            within_range = target_df[
                (abs(query_df.iloc[i]["precursor"] - target_df["precursor"])
                < tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)) 
                & (query_df.iloc[i]["precursor_type"]==target_df["precursor_type"])
                & (target_df["queryID"]!=query_df.iloc[i]["queryID"])
            ]

        else:
            within_range = target_df[
                (abs(query_df.iloc[i]["precursor"] - target_df["precursor"])
                < tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)) 
                & (query_df.iloc[i]["mode"]==target_df["mode"])
                & (target_df["queryID"]!=query_df.iloc[i]["queryID"])
            ]

        within_range.dropna(how='any', inplace=True)

        #catch case where there are no precursor matches
        if within_range.shape[0]==0:
            unmatched+=1
            continue

        #within_range = within_range.sample(frac=1)[:max_rows_per_query]
        within_range.reset_index(inplace=True)
        query_num = query_num+[i for _ in range(len(within_range))]
        seen += len(within_range)

        if seen > printy:

            print(f"{seen} rows created")
            printy = printy + 1e5

        if out is None:
            out = within_range.apply(
                lambda x: add_non_spec_features(query_df.iloc[i], x),
                axis=1,
                result_type="expand"
            )
            
            out.columns = non_spec_columns
            out['query_spec_ID'] = [query_df.iloc[i]["ID"] for x in range(len(within_range))]
            out['target_spec_ID'] = within_range["ID"].tolist()
            out["query"] = [query_df.iloc[i]["spectrum"] for x in range(len(out))]
            out["target"] = within_range["spectrum"]
            out["target_base"] = within_range["inchi_base"]
            out["match"] = (
                query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            )

        else:
            temp = within_range.apply(
                lambda x: add_non_spec_features(query_df.iloc[i], x),
                axis=1,
                result_type="expand"
            )
            
            temp.columns = non_spec_columns
            temp['query_spec_ID'] = [query_df.iloc[i]["ID"] for x in range(len(within_range))]
            temp['target_spec_ID'] =within_range["ID"].tolist()
            temp["query"] = [query_df.iloc[i]["spectrum"] for x in range(len(temp))]
            temp["target"] = within_range["spectrum"]
            temp["target_base"] = within_range["inchi_base"]
            temp["match"] = (
                query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            )
            out = pd.concat([out, temp])

        if len(out) >= max_len:
            
            with open(logpath,'a') as handle:
                handle.write(f'matched prec thresh: {precursor_thresh}, max len:{max_len} adduct match: {adduct_match} in {time.perf_counter()-start}\n')
                handle.write(f'total number of query spectra considered: {seen_}\n')
                handle.write(f'total number of target spectra considered: {seen}\n')
                handle.write(f'total inchicores seen: {len(cores_set)}\n')
                handle.write(f'{unmatched} queries went unmatched\n')
                    
            return out

    if logpath is None:
        
        with open(logpath,'a') as handle:
            handle.write(f'matched prec thresh: {precursor_thresh}, max len:{max_len} adduct match: {adduct_match} in {time.perf_counter()-start}\n')
            handle.write(f'total number of query spectra considered: {seen_}\n')
            handle.write(f'total number of target spectra considered: {seen}\n')
            handle.write(f'total inchicores seen: {len(cores_set)}\n')
            handle.write(f'{unmatched} queries went unmatched\n')

    return out


def create_cleaned_df_chunk(
    input_path,
    output_path,
    sim_methods=None,
    noise_threshes=[0.01],
    centroid_tolerance_vals=[0.05],
    centroid_tolerance_types=["da"],
    reweight_methods=['orig'],
    prec_removes=[True],
    original_order=False
):
    """ """
    # create helper vars
    out_df = None
    spec_columns = [
        "ent_query",
        "npeaks_query",
        "normalent_query",
        "ent_target",
        "npeaks_target",
        "normalent_target",
    ]

    for chunk in os.listdir(input_path):

        if chunk[-3:] != 'pkl':
            continue

        matches_df = pd.read_pickle(f'{input_path}/{chunk}')

        out_df=None
        for remove in prec_removes:

            init_spec_df = matches_df.apply(
                lambda x: get_spec_features(
                    x["query"], x["target"]
                ),
                axis=1,
                result_type="expand",
            )

            init_spec_df.columns = spec_columns

            ticker = 0
            for i in noise_threshes:
                for j in reweight_methods:
                    for k in range(len(centroid_tolerance_vals)):

                        ticker += 1
                        if ticker % 10 == 0:
                            print(f"added {ticker} settings")

                        spec_columns_ = [
                            f"{x}_n:{i} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{j}, pr:{remove}"
                            for x in spec_columns
                        ]


                        sim_columns_ = [
                            f"{x}_n:{i} c:{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]} p:{j}, pr:{remove}"
                            for x in sim_methods
                        ]

                        # clean specs and get corresponding spec features
                        cleaned_df = matches_df.apply(
                            lambda x: clean_and_spec_features(
                                x["query"],
                                x["precquery"],
                                x["target"],
                                x["prectarget"],
                                noise_thresh=i,
                                centroid_thresh=centroid_tolerance_vals[k],
                                centroid_type=centroid_tolerance_types[k],
                                reweight_method=j,
                                prec_remove=remove,
                                original_order=original_order
                            ),
                            axis=1,
                            result_type="expand",
                        ).iloc[:,-2:]
                        
                        cleaned_df.columns = ["query", "target"]

                        if centroid_tolerance_types[k]=='da':
                            clean_matches = cleaned_df.apply(lambda x: tools.match_peaks_in_spectra_separate(
                                    x['query'],
                                    x['target'], 
                                    ms2_da=centroid_tolerance_vals[k]
                                ),
                                axis=1,
                                result_type='expand'
                            )
                        else:
                            clean_matches = cleaned_df.apply(lambda x: tools.match_peaks_in_spectra_separate(
                                    x['query'],
                                    x['target'], 
                                    ms2_ppm=centroid_tolerance_vals[k]
                                ),
                                axis=1,
                                result_type='expand'
                            )
                            
                        clean_matches.columns=[f'mzs_{remove}_{i}_{j}_{k}',f'query_{remove}_{i}_{j}_{k}',f'target_{remove}_{i}_{j}_{k}']

                        out_df = pd.concat(
                            (   
                                out_df,
                                clean_matches,
                            ),
                            axis=1,
                        )

            out_df['precursor'] = (matches_df['precquery'] + matches_df['prectarget'])/2
            out_df['match']=matches_df['match']
            return out_df
    

def distance_sep(
    query,
    target,
    method: str,
    need_normalize_result: bool = True,
) -> float:
    """
    Calculate the distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """

    # Calculate similarity

    if "reverse" in method:
        dist = math_distance.reverse_distance(
            query,
            target,
            metric="_".join(method.split("_")[1:]),
        )

    elif "max" in method:
        dist = math_distance.max_distance(
            query,
            target,
            metric="_".join(method.split("_")[1:]),
        )

    elif "min" in method and method != "minkowski":

        dist = math_distance.min_distance(
            query,
            target,
            metric="_".join(method.split("_")[1:]),
        )

    elif "ave" in method:

        dist = math_distance.ave_distance(
            query,
            target,
            metric="_".join(method.split("_")[1:]),
        )

    else:
        function_name = method + "_distance"
        if hasattr(math_distance, function_name):
            f = getattr(math_distance, function_name)
            dist = f(query, target)

        else:
            raise RuntimeError("Method name: {} error!".format(method))

    # Normalize result
    if need_normalize_result:
        if method not in methods_range:
            try:
                dist_range = methods_range["_".join(method.split("_")[1:])]
            except:
                print(f'error on {method}')
        else:
            dist_range = methods_range[method]

        dist = normalize_distance(dist, dist_range)

    if np.isnan(dist):
        dist=1
        
    return dist

def similarity(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    method: str,
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_normalize_result: bool = True,
) -> float:
    """
    Calculate the similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Similarity between two spectra
    """
    if need_normalize_result:
        return 1 - distance(
            spectrum_query=spectrum_query,
            spectrum_library=spectrum_library,
            method=method,
            need_normalize_result=need_normalize_result,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )
    else:
        return 0 - distance(
            spectrum_query=spectrum_query,
            spectrum_library=spectrum_library,
            method=method,
            need_normalize_result=need_normalize_result,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )
    
def normalize_distance(dist, dist_range):
    if dist_range[1] == np.inf:
        if dist_range[0] == 0:
            result = 1 - 1 / (1 + dist)
        elif dist_range[1] == 1:
            result = 1 - 1 / dist
        else:
            raise NotImplementedError()
    elif dist_range[0] == -np.inf:
        if dist_range[1] == 0:
            result = -1 / (-1 + dist)
        else:
            raise NotImplementedError()
    else:
        result = (dist - dist_range[0]) / (dist_range[1] - dist_range[0])

    if result < 0:
        result = 0.0
    elif result > 1:
        result = 1.0

    return result

def multiple_distance(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    methods: list = None,
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_normalize_result: bool = True,
) -> dict:
    """
    Calculate multiple distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param methods: A list of method names.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """
    if methods is None:
        methods = (
            [i for i in methods_range]
            + [f"reverse_{i}" for i in methods_range]
            + [f"max_{i}" for i in methods_range]
        )

    result = {}
    if ms2_ppm is not None:
        spec_matched = tools.match_peaks_in_spectra(
            spectrum_query, spectrum_library, ms2_ppm=ms2_ppm
        )
    else:
        spec_matched = tools.match_peaks_in_spectra(
            spectrum_query, spectrum_library, ms2_da=ms2_da
        )

    for m in methods:
        dist = distance(
            spec_matched,
            method=m,
            need_normalize_result=need_normalize_result,
        )
        result[m] = float(dist)
    return result

def create_model_and_ind_data(input_path, outputs_path, comparison_metrics):


        df = pd.read_pickle(input_path)

        #for each cleaning setting, we are going to calculate AUCs
        #hang onto old columns so we know what cleaning setting was associated with these AUCs
        ind_aucs_=None
        df_data_gbcs = None
        df_unnorm_dists = None
        for j in range(int(df.shape[1]/3)):

            #jonah...why 3 and not 2?????
            #jonah renaming the columns in sub and not inds appears to be wrong
            # jonah why do we need unormalized distance

            sub = df.iloc[:,(3*j)+1:3*(j+1)] #this corresponds to one cleaned query and target
            old_cols = sub.columns
            sub.columns=['query','target']
            sub['match'] = df['match'].tolist()

            #only need one unnormed setting
            #jonah don't know what the deal is with unnormed
            # if j == 0:
            #     ind_aucs, inds, inds_unnorm = orig_metric_to_df(comparison_metrics, sub, unnnormalized=True)
            #     train_unnorm_dists = pd.concat((train_unnorm_dists,inds_unnorm), axis=1)
            #     train_unnorm_dists_.append(train_unnorm_dists)
            
            ind_aucs, inds = orig_metric_to_df(comparison_metrics, sub)
            ind_aucs['metric'] = [i + '_' + old_cols[0] for i in ind_aucs['metric']]

            ind_aucs_ = pd.concat((ind_aucs_, ind_aucs))
            sub = sub.iloc[:,:2]
            sub.columns=old_cols
            df_data_gbcs = pd.concat((df_data_gbcs,inds), axis=1)

            df_data_gbcs['match'] = df['match'].tolist()

        with open(f'{outputs_path}/model_data.pkl', 'wb') as handle:

            pickle.dump(df_data_gbcs, handle)

        with open(f'{outputs_path}/ind_aucs.pkl', 'wb') as handle:

            pickle.dump(ind_aucs, handle)
