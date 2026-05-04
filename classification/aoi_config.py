from __future__ import annotations

from typing import Any


DEFAULT_SEJONG_BBOX = [126.85, 36.38, 127.15, 36.62]
DEFAULT_GEObOUNDARIES_COLLECTION = "WM/geoLab/geoBoundaries/600/ADM1"
DEFAULT_GEObOUNDARIES_NAME = "Sejong"


def parse_bbox_text(raw: str) -> list[float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox text: {raw!r}. Expected 'west,south,east,north'.")
    bbox = [float(part) for part in parts]
    west, south, east, north = bbox
    if east <= west or north <= south:
        raise ValueError(f"Invalid bbox ordering: {raw!r}")
    return bbox


def resolve_aoi_settings(aoi_asset: str | None, bbox_text: str | None) -> dict[str, Any]:
    asset = (aoi_asset or "").strip()
    if asset:
        return {
            "mode": "asset",
            "asset_id": asset,
            "bbox": list(DEFAULT_SEJONG_BBOX),
            "description": f"EE asset: {asset}",
        }

    if bbox_text and bbox_text.strip():
        bbox = parse_bbox_text(bbox_text)
        return {
            "mode": "bbox",
            "bbox": bbox,
            "description": f"Manual bbox: {bbox}",
        }

    return {
        "mode": "geoboundaries",
        "collection_id": DEFAULT_GEObOUNDARIES_COLLECTION,
        "shape_name": DEFAULT_GEObOUNDARIES_NAME,
        "bbox": list(DEFAULT_SEJONG_BBOX),
        "description": f"geoBoundaries ADM1: {DEFAULT_GEObOUNDARIES_NAME}",
    }


def build_aoi(ee_module, settings: dict[str, Any]):
    mode = settings["mode"]
    if mode == "asset":
        geometry = ee_module.FeatureCollection(settings["asset_id"]).geometry()
    elif mode == "bbox":
        geometry = ee_module.Geometry.Rectangle(settings["bbox"])
    elif mode == "geoboundaries":
        collection = ee_module.FeatureCollection(settings["collection_id"])
        geometry = collection.filter(
            ee_module.Filter.eq("shapeName", settings["shape_name"])
        ).geometry()
    else:
        raise ValueError(f"Unknown AOI mode: {mode}")
    return geometry
