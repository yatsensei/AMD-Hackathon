"""Pluggable wafer classifiers (CNN now; YOLO adapter later).

    from wafer.classifiers import get_classifier
    clf = get_classifier()          # cnn by default ($WAFER_CLASSIFIER)
    clf.classify(wafer_map)
"""
from wafer.classifiers.base import COMMON_RESULT_KEYS, ClassifierBackend, ClassifierNotAvailableError
from wafer.classifiers.cnn import CnnClassifier
from wafer.classifiers.registry import DEFAULT_CLASSIFIER, KNOWN, get_classifier

__all__ = [
    "COMMON_RESULT_KEYS",
    "ClassifierBackend",
    "ClassifierNotAvailableError",
    "CnnClassifier",
    "DEFAULT_CLASSIFIER",
    "KNOWN",
    "get_classifier",
]
