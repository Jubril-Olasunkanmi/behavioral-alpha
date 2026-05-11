import pandas as pd
from src.model import FEATURES

def get_feature_importance(model) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame()
    return pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False)