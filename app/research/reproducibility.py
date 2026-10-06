import hashlib
import json
from pathlib import Path

import pandas as pd


RESEARCH_VERSION = "2.0.0"
VFI_FORMULA_VERSION = "LAZYBEAR_COMPATIBLE_V1"
CYCLE_DEFINITION_VERSION = "ZERO_CROSS_DOWN_TO_ZERO_CROSS_DOWN_V1"
STAGE_DEFINITION_VERSION = "VFI_CYCLE_STAGES_V1"


def dataframe_sha256(data: pd.DataFrame) -> str:
    frame = data.copy().sort_index()
    raw = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def build_run_metadata(
    symbol,
    provider,
    retrieved_at,
    data,
    vfi_parameters,
    extra=None,
):
    metadata = {
        "Research_Version": RESEARCH_VERSION,
        "VFI_Formula_Version": VFI_FORMULA_VERSION,
        "Cycle_Definition_Version": CYCLE_DEFINITION_VERSION,
        "Stage_Definition_Version": STAGE_DEFINITION_VERSION,
        "Symbol": symbol,
        "Provider": provider,
        "Retrieved_At": retrieved_at,
        "Rows": int(len(data)),
        "First_Date": data.index.min().isoformat() if len(data) else None,
        "Last_Date": data.index.max().isoformat() if len(data) else None,
        "Data_SHA256": dataframe_sha256(data),
        "VFI_Parameters": vfi_parameters,
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_run_metadata(path, metadata):
    Path(path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
