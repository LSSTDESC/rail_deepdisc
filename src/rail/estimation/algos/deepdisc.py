import os
import sys
import tempfile

import detectron2.data as d2data
import detectron2.solver as solver
import detectron2.utils.comm as comm
import numpy as np
import qp
from deepdisc.data_format.augment_image import train_augs
from deepdisc.data_format.image_readers import DC2ImageReader
# from deepdisc.data_format.register_data import (register_data_set,
#                                                register_loaded_data_set)
from deepdisc.inference.match_objects import (get_matched_object_classes_new,
                                              get_matched_z_pdfs,
                                              get_matched_z_pdfs_new,
                                              get_matched_z_points_new)
from deepdisc.inference.predictors import return_predictor_transformer
from deepdisc.model.loaders import (RedshiftDictMapper, RedshiftFlatDictMapper,
                                    return_test_loader, return_train_loader)
from deepdisc.model.models import return_lazy_model
from deepdisc.training.trainers import (return_evallosshook,
                                        return_lazy_trainer, return_optimizer,
                                        return_savehook, return_schedulerhook)
from detectron2.config import LazyConfig, get_cfg, instantiate
from detectron2.engine import launch
from detectron2.engine.defaults import create_ddp_model
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import Hdf5Handle, JsonHandle, QPHandle, TableHandle
from rail.estimation.estimator import CatEstimator, CatInformer

from rail.deepdisc.configs import *


def train(config, metadata, train_head=True):
    cfgfile = config["cfgfile"]
    batch_size = config["batch_size"]
    numclasses = config["numclasses"]
    output_dir = config["output_dir"]
    output_name = config["output_name"]
    period = config["period"]
    epoch = config["epoch"]
    head_iters = config["head_iters"]
    full_iters = config["full_iters"]

    cfg = get_lazy_config(cfgfile, batch_size, numclasses)
    cfg_loader = get_loader_config(output_dir, batch_size)

    e1 = epoch * 15
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * 35

    val_per = epoch

    mapper = RedshiftDictMapper(
        DC2ImageReader(), lambda dataset_dict: dataset_dict["filename"]
    ).map_data

    loader = d2data.build_detection_train_loader(
        metadata, mapper=mapper, total_batch_size=batch_size
    )

    if train_head:
        cfg.train.init_checkpoint = None

        model = instantiate(cfg.model)

        for param in model.parameters():
            param.requires_grad = False
        # Phase 1: Unfreeze only the roi_heads
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        # Phase 2: Unfreeze region proposal generator with reduced lr
        for param in model.proposal_generator.parameters():
            param.requires_grad = True

        model.to(cfg.train.device)
        model = create_ddp_model(model, **cfg.train.ddp)

        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.001
        # optimizer = return_optimizer(cfg)
        optimizer = solver.build_optimizer(cfg_loader, model)
        cfg_loader.SOLVER.MAX_ITER = e1  # for DefaultTrainer

        saveHook = return_savehook(output_name)
        # lossHook = return_evallosshook(val_per, model, eval_loader)
        schedulerHook = return_schedulerhook(optimizer)
        # removing the eval loss eval hook for testing as it slows down the training
        # hookList = [lossHook, schedulerHook, saveHook]
        hookList = [schedulerHook, saveHook]

        trainer = return_lazy_trainer(
            model, loader, optimizer, cfg, cfg_loader, hookList
        )

        # how often the trainer prints something to screen, could be a config
        trainer.set_period(period)

        trainer.train(0, head_iters)

        if comm.is_main_process():
            np.save(output_dir + output_name + "_losses", trainer.lossList)
            # np.save(output_dir + output_name + "_val_losses", trainer.vallossList)

        return model

    else:
        cfg.train.init_checkpoint = os.path.join(output_dir, output_name + ".pth")
        cfg_loader.SOLVER.BASE_LR = 0.0001
        cfg_loader.SOLVER.STEPS = [e2, e3]  # do not decay learning rate for retraining
        cfg_loader.SOLVER.MAX_ITER = efinal  # for DefaultTrainer

        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model, **cfg.train.ddp)

        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.0001
        optimizer = solver.build_optimizer(cfg_loader, model)
        cfg_loader.SOLVER.MAX_ITER = efinal  # for DefaultTrainer

        saveHook = return_savehook(output_name)
        # lossHook = return_evallosshook(val_per, model, eval_loader)
        schedulerHook = return_schedulerhook(optimizer)
        # removing the eval loss eval hook for testing as it slows down the training
        # hookList = [lossHook, schedulerHook, saveHook]
        hookList = [schedulerHook, saveHook]

        trainer = return_lazy_trainer(
            model, loader, optimizer, cfg, cfg_loader, hookList
        )

        trainer.set_period(period)
        trainer.train(0, full_iters)

        if comm.is_main_process():
            losses = np.load(output_dir + output_name + "_losses.npy")
            losses = np.concatenate((losses, trainer.lossList))
            np.save(output_dir + output_name + "_losses", losses)

            # vallosses = np.load(output_dir + output_name + "_val_losses.npy")
            # vallosses = np.concatenate((vallosses, trainer.vallossList))
            # np.save(output_dir + output_name + "_val_losses", vallosses)

        return model


class DeepDiscInformer(CatInformer):
    """Placeholder for informer stage class"""

    name = "DeepDiscInformer"
    config_options = CatInformer.config_options.copy()
    # Add defaults and a help message
    # e.g. cfgfile = Param(str, None, required=True,
    #        msg="The primary configuration file for the deepdisc models."),

    # port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    # dist_url="tcp://127.0.0.1:{}".format(port)
    # config_options.update(dist_url=dist_url, machine_rank=0, num_machines=1)

    inputs = [("input", TableHandle), ("metadata", JsonHandle)]
    # outputs = [('model', ModelHandle)]

    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)

    def inform(self, training_data, metadata):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            self.temp_dir = temp_directory_name
            self.set_data("input", training_data)
            self.set_data("metadata", metadata)
            self.run()
            self.finalize()
        return self.get_handle("model")

    def finalize(self):
        pass

    def run(self):
        """
        Train a inception NN on a fraction of the training data
        """
        # train_data = self.get_data("input")
        metadata = self.get_data("metadata")

        print("Caching data")
        # create an iterator here
        itr = self.input_iterator("input")
        for start_idx, _, chunk in itr:
            for idx, image in enumerate(chunk["images"]):
                this_img_metadata = metadata[start_idx + idx]
                image_height = this_img_metadata["height"]
                image_width = this_img_metadata["width"]

                reformed_image = image.reshape(6, image_height, image_width).astype(
                    np.float32
                )

                filename = f"image_{start_idx + idx}.npy"
                file_path = os.path.join(self.temp_dir, filename)
                np.save(file_path, reformed_image)

                # we want the dictionary associated with this particular image.
                metadata[start_idx + idx]["filename"] = file_path

        self.metadata = metadata

        num_gpus = self.config.num_gpus
        num_machines = self.config.num_machines
        machine_rank = self.config.machine_rank

        port = (
            2**15
            + 2**14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
        )
        dist_url = "tcp://127.0.0.1:{}".format(port)

        print("Training head layers")
        train_head = True
        launch(
            train,
            num_gpus,
            num_machines=num_machines,
            machine_rank=machine_rank,
            dist_url=dist_url,
            args=(
                self.config.to_dict(),
                self.metadata,
                train_head,
            ),
        )

        print("Training full model")
        train_head = False
        launch(
            train,
            self.config.num_gpus,
            num_machines=num_machines,
            machine_rank=machine_rank,
            dist_url=dist_url,
            args=(
                self.config.to_dict(),
                self.metadata,
                train_head,
            ),
        )

        #! Question for Grant, what is it that we actually want to save here?
        self.model = dict(nnmodel=self.model)
        self.add_data("model", self.model)


#! I don't think we actually use this class???
class DeepDiscEstimator(CatEstimator):
    """DeepDISC estimator"""

    name = "DeepDiscEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update()

    outputs = [("output", TableHandle)]

    def __init__(self, args, comm=None):
        """Constructor:
        Do Estimator specific initialization"""

        self.nnmodel = None
        CatEstimator.__init__(self, args, comm=comm)

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is not None:
            self.nnmodel = self.model["nnmodel"]

    def run(self):
        test_data = self.get_data("input")

        cfgfile = self.config.cfgfile
        batch_size = self.config.batch_size
        numclasses = self.config.numclasses
        epochs = self.config.epochs
        output_dir = self.config.output_dir
        output_name = self.config.output_name

        cfg = get_lazy_config(cfgfile, batch_size, numclasses)
        cfg_loader = get_loader_config(output_dir, batch_size)

        cfg.train.init_checkpoint = os.path.join(output_dir, output_name) + ".pth"

        # Process test images same way as training set
        predictor = return_predictor_transformer(cfg, cfg_loader)
        mapper = RedshiftFlatDictMapper().map_data

        print("Processing Data")
        dataset_dicts = {}
        dds = []
        for row in test_data:
            dds.append(mapper(row))
        dataset_dicts["test"] = dds

        print("Matching objects")
        true_classes, pred_classes = get_matched_object_classes_new(
            dataset_dicts["test"], predictor
        )
        self.true_zs, self.preds = get_matched_z_points_new(
            dataset_dicts["test"], predictor
        )
        # self.pred = self.preds.squeeze()

    def finalize(self):
        preds = np.array(self.preds)
        self.add_handle("output", data=preds)


class DeepDiscPDFEstimator(CatEstimator):
    """DeepDISC estimator"""

    name = "DeepDiscPDFEstimator"
    config_options = CatInformer.config_options.copy()
    inputs = [("input", TableHandle), ("metadata", JsonHandle)]
    outputs = [("output", QPHandle), ("truth", TableHandle)]

    # config_options.update()
    # config_options.update(hdf5_groupname=SHARED_PARAMS)

    def __init__(self, args, comm=None):
        """Constructor:
        Do Estimator specific initialization"""
        self.nnmodel = None
        CatEstimator.__init__(self, args, comm=comm)
        # self.config.hdf5_groupname = None

    def estimate(self, input_data, metadata):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            self.temp_dir = temp_directory_name
            self.set_data("input", input_data)
            self.set_data("metadata", metadata)
            self.run()
            self.finalize()
        return self.get_handle("output")

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is not None:
            self.nnmodel = self.model["nnmodel"]

    def run(self):
        """
        calculate and return PDFs for each galaxy using the trained flow
        """

        # self.open_model(**self.config)

        # test_data = self.get_data("input")
        metadata = self.get_data("metadata")

        print("caching data")
        itr = self.input_iterator("input")
        for start_idx, _, chunk in itr:
            for idx, image in enumerate(chunk["images"]):
                this_img_metadata = metadata[start_idx + idx]
                image_height = this_img_metadata["height"]
                image_width = this_img_metadata["width"]

                reformed_image = image.reshape(6, image_height, image_width).astype(
                    np.float32
                )

                filename = f"image_{start_idx + idx}.npy"
                file_path = os.path.join(self.temp_dir, filename)
                np.save(file_path, reformed_image)

                # we want the dictionary associated with this particular image.
                metadata[start_idx + idx]["filename"] = file_path

        cfgfile = self.config.cfgfile
        batch_size = self.config.batch_size
        numclasses = self.config.numclasses
        epoch = self.config.epoch
        output_dir = self.config.output_dir
        output_name = self.config.output_name

        cfg = get_lazy_config(cfgfile, batch_size, numclasses)
        cfg_loader = get_loader_config(output_dir, batch_size)
        cfg.train.init_checkpoint = os.path.join(output_dir, output_name) + ".pth"

        self.predictor = return_predictor_transformer(cfg, cfg_loader)

        # Process test images same way as training set
        mapper = RedshiftDictMapper(
            DC2ImageReader(), lambda dataset_dict: dataset_dict["filename"]
        ).map_data

        print("Processing Data")
        #         dataset_dicts = {}
        #         dds = []
        #         for row in test_data:
        #             dds.append(mapper(row))
        #         dataset_dicts["test"] = dds

        print("Matching objects")
        # true_classes, pred_classes = get_matched_object_classes_new(dataset_dicts["test"],  predictor)
        # true_zs, pdfs = get_matched_z_pdfs_new(metadata, self.predictor)
        true_zs, pdfs = get_matched_z_pdfs(
            metadata,
            DC2ImageReader(),
            lambda dataset_dict: dataset_dict["filename"],
            self.predictor,
        )
        self.true_zs = true_zs
        self.pdfs = np.array(pdfs)

    def finalize(self):
        self.zgrid = np.linspace(-1, 5, 200)

        zmode = np.array([self.zgrid[np.argmax(pdf)] for pdf in self.pdfs]).flatten()
        qp_distn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=self.pdfs))
        qp_distn.set_ancil(dict(zmode=zmode))
        qp_distn = self.calculate_point_estimates(qp_distn)
        self.add_handle("output", data=qp_distn)
        truth_dict = dict(redshift=self.true_zs)
        # truth = DS.add_data("truth", truth_dict, TableHandle)
        self.add_handle("truth", data=truth_dict)
