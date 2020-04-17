import matplotlib.pyplot as plt
import cv2
import os
from skimage.morphology import watershed
from scipy import ndimage as ndi
import numpy as np
from shapely.geometry import Polygon
from typing import Union


def IoU(polygon: Polygon, other_polys: Union[Polygon, gpd.GeoDataFrame]):
    """Calculates Intersection over Union
    """
    intersection_areas = other_polys.intersection(polygon).area
    union_areas = polygon.area + other_polys.area - intersection_areas
    return intersection_areas / union_areas


def filter_prediction(pred_mask: np.ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    opening = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closing


def fit_shapes(pred_mask: np.ndarray, use_shape_thesh=0.7):
    contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fitted_image = np.zeros(pred_mask.shape)
    for cont in contours:
        _, trig_vertices = cv2.minEnclosingTriangle(cont)
        rect_params = cv2.minAreaRect(cont)
        try:
            trig_vertices = trig_vertices.squeeze().astype(np.int32)
            rect_vertices = cv2.boxPoints(rect_params)
            rect_vertices = np.int32(rect_vertices)

            cont_poly = Polygon(cont.squeeze())
            rect_poly = Polygon(rect_vertices)
            trig_poly = Polygon(trig_vertices)

            cont_poly = cont_poly.buffer(0)
            rect_poly = rect_poly.buffer(0)
            trig_poly = trig_poly.buffer(0)

            rect_score = IoU(cont_poly, rect_poly)
            trig_score = IoU(cont_poly, trig_poly)
        except:
            continue

        use_shape = (trig_score > use_shape_thesh) or (rect_score > use_shape_thesh)
        if use_shape:
            if rect_score > trig_score:
                cv2.drawContours(fitted_image, [rect_vertices], -1, (255,0,0), -1)
            else:
                cv2.drawContours(fitted_image, [trig_vertices], -1, (255,0,0), -1)
        else:
            cv2.drawContours(fitted_image, [cont.squeeze()], -1, (255,0,0), -1)
    return fitted_image


def postprocess(pred_mask: np.ndarray, threshold=0.5):
    pred_mask = pred_mask > threshold
#     pred_mask = filter_prediction(pred_mask)
    if pred_mask.sum() != 0:
        pred_mask = fit_shapes(pred_mask)
    return pred_mask
