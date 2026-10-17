"""Pluggable wafer classifiers (CNN default; experimental YOLO-cls optional).

    from wafer.classifiers import get_classifier
    clf = get_classifier()          # cnn by default ($WAFER_CLASSIFIER)
    clf.classify(wafer_map)
    # get_classifier("yolo") → experimental YOLO-cls, or ClassifierNotAvailableError
"""
from wafer.classifiers.base import COMMON_RESULT_KEYS, ClassifierBackend, ClassifierNotAvailableError
from wafer.classifiers.cnn import CnnClassifier
from wafer.classifiers.registry import DEFAULT_CLASSIFIER, KNOWN, get_classifier
from wafer.classifiers.yolo import YoloClassifier

__all__ = [
    "COMMON_RESULT_KEYS",
    "ClassifierBackend",
    "ClassifierNotAvailableError",
    "CnnClassifier",
    "YoloClassifier",
    "DEFAULT_CLASSIFIER",
    "KNOWN",
    "get_classifier",
]
