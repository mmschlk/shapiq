import copy
import struct
from typing import Union

import numpy as np
import scipy
from explainer.tree.base import TreeModel
from packaging import version

from shapiq.utils import safe_isinstance

try:
    from xgboost.sklearn import XGBClassifier, XGBRegressor
except ImportError:
    pass


def convert_xgboost_trees(
    tree_model: Union["XGBClassifier", "XGBRegressor"],
) -> list[TreeModel]:
    if safe_isinstance(tree_model, "xgboost.sklearn.XGBRegressor"):
        model = tree_model.get_booster()
        xgb_loader = XGBTreeModelLoader(model)
        trees = xgb_loader.get_trees()
        return copy.deepcopy(trees)
    if safe_isinstance(tree_model, "xgboost.sklearn.XGBClassifier"):
        raise NotImplementedError("XGBoost Classifier not implemented yet.")


class XGBTreeModelLoader:
    """This loads an XGBoost model directly from a raw memory dump.

    We can't use the JSON dump because due to numerical precision issues those
    tree can actually be wrong when feature values land almost on a threshold.
    """

    def __init__(self, xgb_model):
        # new in XGBoost 1.1, 'binf' is appended to the buffer
        self.buf = xgb_model.save_raw()
        if self.buf.startswith(b"binf"):
            self.buf = self.buf[4:]
        self.pos = 0

        # load the model parameters
        self.base_score = self.read("f")
        self.num_feature = self.read("I")
        self.num_class = self.read("i")
        self.contain_extra_attrs = self.read("i")
        self.contain_eval_metrics = self.read("i")
        self.read_arr("i", 29)  # reserved
        self.name_obj_len = self.read("Q")
        self.name_obj = self.read_str(self.name_obj_len)
        self.name_gbm_len = self.read("Q")
        self.name_gbm = self.read_str(self.name_gbm_len)

        # new in XGBoost 1.0 is that the base_score is saved untransformed (https://github.com/dmlc/xgboost/pull/5101)
        # so we have to transform it depending on the objective
        import xgboost

        if version.parse(xgboost.__version__).major >= 1:
            if self.name_obj in ["binary:logistic", "reg:logistic"]:
                self.base_score = scipy.special.logit(self.base_score)  # pylint: disable=no-member

        assert self.name_gbm == "gbtree", (
            "Only the 'gbtree' model type is supported, not '%s'!" % self.name_gbm
        )

        # load the gbtree specific parameters
        self.num_trees = self.read("i")
        self.num_roots = self.read("i")
        self.num_feature = self.read("i")
        self.pad_32bit = self.read("i")
        self.num_pbuffer_deprecated = self.read("Q")
        self.num_output_group = self.read("i")
        self.size_leaf_vector = self.read("i")
        self.read_arr("i", 32)  # reserved

        # load each tree
        self.num_roots = np.zeros(self.num_trees, dtype=np.int32)
        self.num_nodes = np.zeros(self.num_trees, dtype=np.int32)
        self.num_deleted = np.zeros(self.num_trees, dtype=np.int32)
        self.max_depth = np.zeros(self.num_trees, dtype=np.int32)
        self.num_feature = np.zeros(self.num_trees, dtype=np.int32)
        self.size_leaf_vector = np.zeros(self.num_trees, dtype=np.int32)
        self.node_parents = []
        self.node_cleft = []
        self.node_cright = []
        self.node_sindex = []
        self.node_info = []
        self.loss_chg = []
        self.sum_hess = []
        self.base_weight = []
        self.leaf_child_cnt = []
        for i in range(self.num_trees):
            # load the per-tree params
            self.num_roots[i] = self.read("i")
            self.num_nodes[i] = self.read("i")
            self.num_deleted[i] = self.read("i")
            self.max_depth[i] = self.read("i")
            self.num_feature[i] = self.read("i")
            self.size_leaf_vector[i] = self.read("i")

            # load the nodes
            self.read_arr("i", 31)  # reserved
            self.node_parents.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_cleft.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_cright.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_sindex.append(np.zeros(self.num_nodes[i], dtype=np.uint32))
            self.node_info.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            for j in range(self.num_nodes[i]):
                self.node_parents[-1][j] = self.read("i")
                self.node_cleft[-1][j] = self.read("i")
                self.node_cright[-1][j] = self.read("i")
                self.node_sindex[-1][j] = self.read("I")
                self.node_info[-1][j] = self.read("f")

            # load the stat nodes
            self.loss_chg.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.sum_hess.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.base_weight.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.leaf_child_cnt.append(np.zeros(self.num_nodes[i], dtype=int))
            for j in range(self.num_nodes[i]):
                self.loss_chg[-1][j] = self.read("f")
                self.sum_hess[-1][j] = self.read("f")
                self.base_weight[-1][j] = self.read("f")
                self.leaf_child_cnt[-1][j] = self.read("i")

    def get_trees(self, data=None, data_missing=None):
        shape = (self.num_trees, self.num_nodes.max())
        self.children_default = np.zeros(shape, dtype=int)
        self.features = np.zeros(shape, dtype=int)
        self.thresholds = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
        trees = []
        for i in range(self.num_trees):
            for j in range(self.num_nodes[i]):
                if np.right_shift(self.node_sindex[i][j], np.uint32(31)) != 0:
                    self.children_default[i, j] = self.node_cleft[i][j]
                else:
                    self.children_default[i, j] = self.node_cright[i][j]
                self.features[i, j] = self.node_sindex[i][j] & (
                    (np.uint32(1) << np.uint32(31)) - np.uint32(1)
                )
                if self.node_cleft[i][j] >= 0:
                    # Xgboost uses < for thresholds where shap uses <=
                    # Move the threshold down by the smallest possible increment
                    self.thresholds[i, j] = np.nextafter(self.node_info[i][j], -np.float32(np.inf))
                else:
                    self.values[i, j] = self.node_info[i][j]

            size = len(self.node_cleft[i])
            trees.append(
                TreeModel(
                    children_left=self.node_cleft[i],
                    children_right=self.node_cright[i],
                    features=self.features[i, :size],
                    thresholds=self.thresholds[i, :size],
                    values=self.values[i, :size].flatten(),
                    node_sample_weight=self.sum_hess[i],
                )
            )
        return trees

    def read(self, dtype):
        size = struct.calcsize(dtype)
        val = struct.unpack(dtype, self.buf[self.pos : self.pos + size])[0]
        self.pos += size
        return val

    def read_arr(self, dtype, n_items):
        format = "%d%s" % (n_items, dtype)
        size = struct.calcsize(format)
        val = struct.unpack(format, self.buf[self.pos : self.pos + size])[0]
        self.pos += size
        return val

    def read_str(self, size):
        val = self.buf[self.pos : self.pos + size].decode("utf-8")
        self.pos += size
        return val

    def print_info(self):
        print("--- global parmeters ---")
        print("base_score =", self.base_score)
        print("num_feature =", self.num_feature)
        print("num_class =", self.num_class)
        print("contain_extra_attrs =", self.contain_extra_attrs)
        print("contain_eval_metrics =", self.contain_eval_metrics)
        print("name_obj_len =", self.name_obj_len)
        print("name_obj =", self.name_obj)
        print("name_gbm_len =", self.name_gbm_len)
        print("name_gbm =", self.name_gbm)
        print()
        print("--- gbtree specific parameters ---")
        print("num_trees =", self.num_trees)
        print("num_roots =", self.num_roots)
        print("num_feature =", self.num_feature)
        print("pad_32bit =", self.pad_32bit)
        print("num_pbuffer_deprecated =", self.num_pbuffer_deprecated)
        print("num_output_group =", self.num_output_group)
        print("size_leaf_vector =", self.size_leaf_vector)
