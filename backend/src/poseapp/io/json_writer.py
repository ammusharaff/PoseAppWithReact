# src/poseapp/io/json_writer.py
import json  # standard JSON serialization module
from typing import Any  # for flexible type hints (any data structure)

def write_keypoints_json(path: str, data: Any):  # function to write data as formatted JSON
    with open(path, "w", encoding="utf-8") as f:  # open file for writing (UTF-8 to support all characters)
        json.dump(data, f, ensure_ascii=False, indent=2)  # serialize `data` to JSON, keeping Unicode and pretty format
