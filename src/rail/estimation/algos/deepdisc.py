import detectron2.data as d2data
import detectron2.solver as solver
import numpy as np
import qp
from deepdisc.data_format.augment_image import train_augs
from deepdisc.data_format.image_readers import DC2ImageReader
# from deepdisc.data_format.register_data import (register_data_set,
#                                                register_loaded_data_set)
from deepdisc.inference.match_objects import (get_matched_object_classes_new,
                                              get_matched_z_pdfs_new,
                                              get_matched_z_points_new)
from deepdisc.inference.predictors import return_predictor_transformer
from deepdisc.model.loaders import (RedshiftDictMapper, RedshiftFlatDictMapper,
                                    return_test_loader, return_train_loader)
from deepdisc.model.models import return_lazy_model
from deepdisc.training.trainers import (return_evallosshook,
                                        return_lazy_trainer, return_optimizer,
                                        return_savehook, return_schedulerhook)
from detectron2.config import LazyConfig, get_cfg
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import TableHandle
from rail.estimation.estimator import CatEstimator, CatInformer

from rail.deepdisc.configs import *


class DeepDiscDictInformer(CatInformer):
    """Placeholder for informer stage class"""

    name = "DeepDiscDictInformer"
    config_options = CatInformer.config_options.copy()

    inputs = [("images", TableHandle), ("metadata", TableHandle)]

    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)

    def inform(self, images, metadata):
        """The main interface method for Informers

        This will attach the input_data to this `Informer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the model that it creates to this Estimator
        by using `self.add_data('model', model)`.

        Finally, this will return a ModelHandle providing access to the trained model.

        Parameters
        ----------
        input_data : `dict` or `TableHandle`
            dictionary of all input data, or a `TableHandle` providing access to it

        Returns
        -------
        model : ModelHandle
            Handle providing access to trained model
        """
        self.set_data("images", images)
        self.set_data("metadata", metadata)

        self.run()
        self.finalize()
        return self.get_handle("model")

    def run(self):
        """
        Train a inception NN on a fraction of the training data
        """
        train_data = self.get_data("images")
        metadata = self.get_data("metadata")

        print(metadata)

        cfgfile = self.config.cfgfile
        batch_size = self.config.batch_size
        numclasses = self.config.numclasses
        epochs = self.config.epochs
        output_dir = self.config.output_dir
        output_name = self.config.output_name

        val_per = 5

        cfg = get_lazy_config(cfgfile, batch_size, numclasses)
        cfg_loader = get_loader_config(output_dir, batch_size, epochs)

        model = return_lazy_model(cfg)
        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.001
        # optimizer = return_optimizer(cfg)
        optimizer = solver.build_optimizer(cfg_loader, model)

        """
        When using the single test dictionary, add this code and replace "mapper" below
        
        
        def dc2_key_mapper(dataset_dict):
            filename = dataset_dict["filename"]
            return filename

        IR = DC2ImageReader()
        mapper = RedshiftDictMapper(IR, dc2_key_mapper).map_data
        """

        mapper = RedshiftFlatDictMapper().map_data
        loader = d2data.build_detection_train_loader(
            train_data, mapper=mapper, total_batch_size=batch_size
        )
        test_loader = d2data.build_detection_test_loader(
            train_data, mapper=mapper, batch_size=batch_size
        )

        saveHook = return_savehook(output_name)
        lossHook = return_evallosshook(val_per, model, test_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(
            model, loader, optimizer, cfg, cfg_loader, hookList
        )

        trainer.set_period(5)

        print("Model training:")
        trainer.train(0, epochs)

        self.model = dict(nnmodel=model)
        self.add_data("model", self.model)


class DeepDiscInformer(CatInformer):
    """Placeholder for informer stage class"""

    name = "DeepDiscInformer"
    config_options = CatInformer.config_options.copy()
    # Add defaults and a help message
    # e.g. cfgfile = Param(str, None, required=True,
    #        msg="The primary configuration file for the deepdisc models."),

    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)

    def run(self):
        """
        Train a inception NN on a fraction of the training data
        """
        train_data = self.get_data("input")

        cfgfile = self.config.cfgfile
        batch_size = self.config.batch_size
        numclasses = self.config.numclasses
        epochs = self.config.epochs
        output_dir = self.config.output_dir
        output_name = self.config.output_name

        val_per = 5

        cfg = get_lazy_config(cfgfile, batch_size, numclasses)
        cfg_loader = get_loader_config(output_dir, batch_size, epochs)

        model = return_lazy_model(cfg)
        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.001
        # optimizer = return_optimizer(cfg)
        optimizer = solver.build_optimizer(cfg_loader, model)

        """
        When using the single test dictionary, add this code and replace "mapper" below
        
        
        def dc2_key_mapper(dataset_dict):
            filename = dataset_dict["filename"]
            return filename

        IR = DC2ImageReader()
        mapper = RedshiftDictMapper(IR, dc2_key_mapper).map_data
        """

        mapper = RedshiftFlatDictMapper().map_data
        loader = d2data.build_detection_train_loader(
            train_data, mapper=mapper, total_batch_size=batch_size
        )
        test_loader = d2data.build_detection_test_loader(
            train_data, mapper=mapper, batch_size=batch_size
        )

        saveHook = return_savehook(output_name)
        lossHook = return_evallosshook(val_per, model, test_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(
            model, loader, optimizer, cfg, cfg_loader, hookList
        )

        trainer.set_period(5)

        print("Model training:")
        trainer.train(0, epochs)

        self.model = dict(nnmodel=model)
        self.add_data("model", self.model)


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
        cfg_loader = get_loader_config(output_dir, batch_size, epochs)

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
    config_options = CatEstimator.config_options.copy()
    config_options.update()
    # config_options.update(hdf5_groupname=SHARED_PARAMS)

    def __init__(self, args, comm=None):
        """Constructor:
        Do Estimator specific initialization"""
        self.nnmodel = None
        CatEstimator.__init__(self, args, comm=comm)
        # self.config.hdf5_groupname = None

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is not None:
            self.nnmodel = self.model["nnmodel"]

    def run(self):
        """
        calculate and return PDFs for each galaxy using the trained flow
        """

        # self.open_model(**self.config)

        test_data = self.get_data("input")

        cfgfile = self.config.cfgfile
        batch_size = self.config.batch_size
        numclasses = self.config.numclasses
        epochs = self.config.epochs
        output_dir = self.config.output_dir
        output_name = self.config.output_name

        cfg = get_lazy_config(cfgfile, batch_size, numclasses)
        cfg_loader = get_loader_config(output_dir, batch_size, epochs)
        cfg.train.init_checkpoint = os.path.join(output_dir, output_name) + ".pth"

        self.predictor = return_predictor_transformer(cfg, cfg_loader)

        # Process test images same way as training set
        mapper = RedshiftFlatDictMapper().map_data

        print("Processing Data")
        dataset_dicts = {}
        dds = []
        for row in test_data:
            dds.append(mapper(row))
        dataset_dicts["test"] = dds

        self.zgrid = np.linspace(-1, 5, 200)

        print("Matching objects")
        # true_classes, pred_classes = get_matched_object_classes_new(dataset_dicts["test"],  predictor)
        true_zs, pdfs = get_matched_z_pdfs_new(dataset_dicts["test"], self.predictor)
        self.pdfs = np.array(pdfs)

    def finalize(self):
        zmode = np.array([self.zgrid[np.argmax(pdf)] for pdf in self.pdfs]).flatten()
        qp_distn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=self.pdfs))
        qp_distn.set_ancil(dict(zmode=zmode))
        qp_distn = self.calculate_point_estimates(qp_distn)
        self.add_handle("output", data=qp_distn)
