# first line: 1
@memory.cache  # Cache the preprocessor
def build_enhanced_pipeline(stopwords_list):
    """
    Create a pipeline that:
    1) Transforms text via TF–IDF (for preprocessed text columns)
    2) Extracts + scales numeric features
    3) Oversamples with SMOTE
    4) Uses a VotingClassifier ensemble (RF, LightGBM, LR)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('captions_tfidf', 
             TfidfVectorizer(
                 stop_words=stopwords_list,
                 max_features=2000,    # Reduced for speed
                 ngram_range=(1, 1),   
                 min_df=2,
                 max_df=0.95
             ), 
             'captions_clean'),
            
            ('bio_tfidf', 
             TfidfVectorizer(
                 stop_words=stopwords_list,
                 max_features=2000,   
                 ngram_range=(1, 1),  
                 min_df=2,
                 max_df=0.95
             ), 
             'biography_clean'),
            
            ('numeric', 
             Pipeline([
                 ('feat_eng', FunctionTransformer(extract_numeric_features)),
                 ('scaler', MinMaxScaler())
             ]), 
             ['follower_count', 'following_count', 'post_count']
             # Add 'account_age_days' here if available
            )
        ],
        remainder='drop'
    )
    
    # Initialize classifiers
    rf = RandomForestClassifier(
        n_estimators=100,    
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Switch to LightGBM for faster gradient boosting
    gb = lgb.LGBMClassifier(
        n_estimators=100,     
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    lr = LogisticRegression(
        C=2.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft',
        n_jobs=-1  # Enable parallelism if supported
    )
    
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy="auto")),
        ('classifier', ensemble)
    ], memory=memory)
    
    return pipeline
