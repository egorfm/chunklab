import math
from typing import Tuple, Optional


def compute_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return width * height


def compute_bbox_iou(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float]
) -> float:
    """Computes Intersection over Union (IoU) of two 2D bounding boxes."""
    x0_a, y0_a, x1_a, y1_a = bbox_a
    x0_b, y0_b, x1_b, y1_b = bbox_b

    inter_x0 = max(x0_a, x0_b)
    inter_y0 = max(y0_a, y0_b)
    inter_x1 = min(x1_a, x1_b)
    inter_y1 = min(y1_a, y1_b)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area_a = compute_bbox_area(bbox_a)
    area_b = compute_bbox_area(bbox_b)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_bbox_containment(
    chunk_bbox: Tuple[float, float, float, float],
    target_bbox: Tuple[float, float, float, float]
) -> float:
    """Computes fraction of target bounding box covered by chunk bounding box."""
    x0_a, y0_a, x1_a, y1_a = chunk_bbox
    x0_b, y0_b, x1_b, y1_b = target_bbox

    inter_x0 = max(x0_a, x0_b)
    inter_y0 = max(y0_a, y0_b)
    inter_x1 = min(x1_a, x1_b)
    inter_y1 = min(y1_a, y1_b)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    target_area = compute_bbox_area(target_bbox)
    if target_area <= 0:
        return 0.0
    return inter_area / target_area


def check_relevance(
    retrieved_page: int,
    retrieved_bbox: Tuple[float, float, float, float],
    retrieved_text: str,
    target_page: int,
    target_bbox: Tuple[float, float, float, float],
    target_text: Optional[str] = None,
    iou_threshold: float = 0.20,
    containment_threshold: float = 0.40
) -> Tuple[bool, float]:
    """Determines whether a retrieved chunk is relevant to the ground truth target.

    Matches by page number and 2D coordinates (IoU or coverage containment).
    Falls back to text containment if coordinates are absent (0.0).
    """
    # 1. Page number must match
    if retrieved_page != target_page:
        return False, 0.0

    target_has_bbox = any(c > 0.0 for c in target_bbox)
    retrieved_has_bbox = any(c > 0.0 for c in retrieved_bbox)

    if target_has_bbox and retrieved_has_bbox:
        iou = compute_bbox_iou(retrieved_bbox, target_bbox)
        containment = compute_bbox_containment(retrieved_bbox, target_bbox)
        
        # Consider relevant if either IoU meets threshold or chunk covers target fragment
        is_match = (iou >= iou_threshold) or (containment >= containment_threshold)
        match_score = max(iou, containment)
        return is_match, round(match_score, 4)

    # 2. Text containment fallback if coordinates are missing/zeroed
    if target_text and target_text.strip():
        clean_target = target_text.strip().lower()
        clean_retrieved = retrieved_text.strip().lower()
        if clean_target in clean_retrieved or clean_retrieved in clean_target:
            return True, 1.0

    return False, 0.0
