"""Pure, credential-free learning evidence. Does not place orders or access storage."""
import hashlib
import math
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd


def finite_scalar(value):
    if isinstance(value, (int, float, np.integer, np.floating, np.bool_)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def json_value(value):
    if value is None or isinstance(value, (str, bool)): return value
    if isinstance(value, dict): return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)): return value.isoformat()
    return finite_scalar(value)


def capture_features(frame, feature_names, model_sha, schema, model_id=None):
    """Preserve exact raw numeric features, explicit nulls and the available history."""
    names = list(feature_names)
    columns = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    columns = list(dict.fromkeys(columns + names))
    tail = frame.tail(len(names))  # Bounded by the configured schema, no extra trading setting.
    values = tail.reindex(columns=columns).to_numpy(dtype=float, copy=True)
    values = np.where(np.isfinite(values), values, None).tolist()
    availability = tail["datetime_dt"].astype(str).tolist() if "datetime_dt" in tail else [""] * len(tail)
    history = [{"bar_start": str(index), "available_at": at, "values": dict(zip(columns, row))}
               for index, at, row in zip(tail.index, availability, values)]
    latest = history[-1]["values"] if history else {c: None for c in columns}
    missing = [c for c in names if latest.get(c) is None]
    return {
        "event_id": uuid4().hex,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source": "OBSERVED_PRE_INFERENCE_FEATURE_FRAME",
        "model_sha256": model_sha,
        "model_id": model_id,
        "feature_schema": schema,
        "feature_names": names,
        "feature_names_sha256": hashlib.sha256("|".join(names).encode()).hexdigest(),
        "features": dict(latest),
        "feature_history": history,
        "feature_complete": bool(names) and not missing,
        "missing_or_nonfinite_features": missing,
        "history_rows": len(history),
        "raw_class_probabilities": None,
        "actual_model_input": None,
    }


def execution_evidence(data):
    """Whitelist execution fields; never copy account/token/raw protocol payloads."""
    nested = data.get("values", {}) if isinstance(data.get("values"), dict) else {}
    def get(*keys):
        for key in keys:
            value = data.get(key, nested.get(key))
            if value is not None and str(value).strip(): return value
        return None
    order = get("order_no", "ord_no", "odno", "orgn_odno", "ord_no1", "1")
    def number(*keys):
        value = get(*keys)
        try: return finite_scalar(float(str(value).replace(",", "")))
        except (TypeError, ValueError): return None
    return {
        "source": "OBSERVED_BROKER_EXECUTION_MESSAGE",
        "order_reference_sha256": hashlib.sha256(str(order).encode()).hexdigest() if order else None,
        "symbol": str(get("symbol", "stk_cd", "jmcode", "item") or ""),
        "incremental_quantity": number("filled_qty", "che_qty", "ft_ccld_qty", "ccld_qty", "11"),
        "cumulative_quantity": number("tot_che_qty", "ft_tot_ccld_qty", "acc_filled_qty", "14"),
        "remaining_quantity": number("nccs_qty", "ft_rem_qty", "unfilled_qty", "13"),
        "reported_fill_price": number("filled_price", "execution_price", "ccld_unpr", "ft_ccld_unpr", "che_price"),
        "reported_average_fill_price": number("avg_fill_price", "avg_ccld_unpr"),
        "reported_fee_usd": number("fee_usd", "commission_usd"),
        "reported_execution_time": str(get("execution_time", "ccld_time", "che_time", "ft_ccld_tmd") or ""),
        "price_semantics": "EXPLICIT_EXECUTION_FIELDS_ONLY_UNKNOWN_IF_NULL",
    }
