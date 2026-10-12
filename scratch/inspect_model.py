import joblib
from pathlib import Path

model_paths = [
    'c:/Users/chabo/OneDrive/바탕 화면/lumos/models/model_main_data_refresh.pkl',
    'c:/Users/chabo/OneDrive/바탕 화면/lumos/models/model_champion.pkl'
]

for mp in model_paths:
    print(f"--- Inspecting {Path(mp).name} ---")
    try:
        model = joblib.load(mp)
        
        # Check if it's a pipeline or a direct model
        if hasattr(model, 'steps'):
            print("Model is a Pipeline.")
            classifier = model.steps[-1][1]
        else:
            classifier = model
            
        if hasattr(classifier, 'feature_name_'):
            features = classifier.feature_name_
        elif hasattr(classifier, 'feature_names_in_'):
            features = classifier.feature_names_in_
        elif hasattr(classifier, 'feature_name'):
            features = classifier.feature_name()
        else:
            features = []
            
        print(f"Model type: {type(classifier)}")
        print(f"Number of features: {len(features)}")
        if len(features) > 0:
            soxl_count = sum(1 for f in features if "SOXL" in str(f).upper())
            soxs_count = sum(1 for f in features if "SOXS" in str(f).upper())
            qqq_count = sum(1 for f in features if "QQQ" in str(f).upper())
            print(f"Contains SOXL features: {soxl_count}")
            print(f"Contains SOXS features: {soxs_count}")
            print(f"Contains QQQ features: {qqq_count}")
        else:
            print("Feature names not available directly.")
            
        if hasattr(classifier, 'get_params'):
            print(f"Params: {classifier.get_params()}")
            
    except Exception as e:
        print(f"Error loading {mp}: {e}")
