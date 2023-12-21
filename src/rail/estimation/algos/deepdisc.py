import json
import os
import sys
import tempfile
from fvcore.common.checkpoint import Checkpointer
import detectron2.data as d2data
import detectron2.solver as solver
import detectron2.utils.comm as comm
import numpy as np
import qp
from ceci.config import StageParameter as Param
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
from detectron2.config import LazyConfig, get_cfg, instantiate, CfgNode
from detectron2.engine import launch
from detectron2.engine.defaults import create_ddp_model
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import Hdf5Handle, JsonHandle, ModelHandle, QPHandle, TableHandle
from rail.estimation.estimator import CatEstimator, CatInformer


def train(config, all_metadata, train_head=True):
    
    cfgfile = config["cfgfile"]
    batch_size = config["batch_size"]
    output_dir = config["output_dir"]
    run_name = config["run_name"]
    print_frequency = config["print_frequency"]
    epoch = config["epoch"]
    head_epochs = config["head_epochs"]
    full_epochs = config["full_epochs"]
    training_percent = config["training_percent"]


    e1 = epoch * head_epochs
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * full_epochs

    val_per = epoch

    # Create slices for the input data
    total_images = len(all_metadata)
    split_index = int(np.floor(total_images * training_percent))
    train_slice = slice(split_index)
    eval_slice = slice(split_index, total_images)

    mapper = RedshiftDictMapper(
        DC2ImageReader(), lambda dataset_dict: dataset_dict["filename"]
    ).map_data

    training_loader = d2data.build_detection_train_loader(
        all_metadata[train_slice], mapper=mapper, total_batch_size=batch_size
    )

    eval_loader = d2data.build_detection_test_loader(
        all_metadata[eval_slice], mapper=mapper, batch_size=batch_size
    )
    
    
    cfg = LazyConfig.load(cfgfile)
    cfg.OUTPUT_DIR = output_dir

    
    model = return_lazy_model(cfg, train_head)
    cfg.optimizer.params.model = model
    optimizer = solver.build_optimizer(cfg, model)


    saveHook = return_savehook(run_name)
    lossHook = return_evallosshook(val_per, model, eval_loader)
    schedulerHook = return_schedulerhook(optimizer)
    #hookList = [lossHook, schedulerHook, saveHook]
    hookList = [schedulerHook, saveHook]

    if train_head:

        cfg.SOLVER.MAX_ITER = e1  # for DefaultTrainer

        trainer = return_lazy_trainer(
            model, training_loader, optimizer, cfg, hookList
        )

        trainer.set_period(print_frequency)

        trainer.train(0, e1)

        if comm.is_main_process():
            np.save(output_dir + run_name + "_losses", trainer.lossList)
            # np.save(output_dir + run_name + "_val_losses", trainer.vallossList)



    else:
        cfg.train.init_checkpoint = os.path.join(output_dir, run_name + ".pth")
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.MAX_ITER = efinal  # for DefaultTrainer

        cfg.optimizer.lr = 0.0001


        trainer = return_lazy_trainer(
            model, training_loader, optimizer, cfg, hookList
        )

        trainer.set_period(print_frequency)
        trainer.train(0, efinal)

        if comm.is_main_process():
            losses = np.load(output_dir + run_name + "_losses.npy")
            losses = np.concatenate((losses, trainer.lossList))
            np.save(output_dir + run_name + "_losses", losses)

            # vallosses = np.load(output_dir + run_name + "_val_losses.npy")
            # vallosses = np.concatenate((vallosses, trainer.vallossList))
            # np.save(output_dir + run_name + "_val_losses", vallosses)




class DeepDiscInformer(CatInformer):
    """Placeholder for informer stage class"""

    name = "DeepDiscInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(
        cfgfile=Param(str, None, required=True, msg="The primary configuration file for the deepdisc models."),
        batch_size=Param(int, 1, required=False, msg="Batch size of data to load."),
        epoch=Param(int, 20, required=False, msg="Number of iterations per epooch."),
        output_dir=Param(str, "./", required=False, msg="The directory to write output to."),
        run_name=Param(str, "run", required=False, msg="Name of the training run."),
        chunk_size=Param(int, 100, required=False, msg="Chunk size used within detectron2 code."),
        training_percent=Param(float, 0.8, required=False, msg="The fraction of input data used to split into training/evaluation sets"),
        num_camera_filters=Param(int, 6, required=False, msg="The number of camera filters for the dataset used (LSST has 6)."),
        print_frequency=Param(int, 5, required=False, msg="How often to print in-progress output (happens every x number of iterations)."),
        head_epochs=Param(int, 0, required=False, msg="How many iterations when training the head layers (while the backbone layers are frozen)."),
        full_epochs=Param(int, 0, required=False, msg="How many iterations when training the head layers and unfrozen backbone layers together."),
        num_gpus=Param(int, 4, required=False, msg="Number of processes per machine. When using GPUs, this should be the number of GPUs."),
        num_machines=Param(int, 1, required=False, msg="The total number of machines."),
        machine_rank=Param(int, 0, required=False, msg="The rank of this machine."),
    )
    inputs = [('input', TableHandle), ('metadata', JsonHandle)]

    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)

    def inform(self, input_data, input_metadata):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            self.temp_dir = temp_directory_name
            self.set_data("input", input_data)
            self.set_data("metadata", input_metadata)
            self.run()
            self.finalize()
        return self.get_handle("model")

    def run(self):
        """
        Train a inception NN on a fraction of the training data
        """
        self.metadata = self.get_data("metadata")

        print("Caching data")
        flattened_image_iterator = self.input_iterator("input")
        for start_idx, _, images in flattened_image_iterator:
            for image_idx, image in enumerate(images["images"]):
                this_image_metadata = self.metadata[start_idx + image_idx]
                image_height = this_image_metadata["height"]
                image_width = this_image_metadata["width"]

                reformed_image = image.reshape(
                    self.config.num_camera_filters, image_height, image_width
                ).astype(np.float32)

                filename = f"image_{start_idx + image_idx}.npy"
                file_path = os.path.join(self.temp_dir, filename)
                np.save(file_path, reformed_image)

                this_image_metadata["filename"] = file_path

        dist_url = self._get_dist_url()

        print("Training head layers")
        train_head = True
        launch(
            train,
            num_gpus_per_machine=self.config.num_gpus,
            num_machines=self.config.num_machines,
            machine_rank=self.config.machine_rank,
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
            num_gpus_per_machine=self.config.num_gpus,
            num_machines=self.config.num_machines,
            machine_rank=self.config.machine_rank,
            dist_url=dist_url,
            args=(
                self.config.to_dict(),
                self.metadata,
                train_head,
            ),
        )

        cfg = LazyConfig.load(self.config.cfgfile)
        model = instantiate(cfg.model)
        weights_file_path = os.path.join(self.config.output_dir, self.config.run_name + ".pth")
        fv_cp = Checkpointer(model, self.config.output_dir)
        weights = fv_cp._load_file(weights_file_path)

        self.model = dict(nnmodel=weights)
        self.add_data("model", self.model)

    def _get_dist_url(self):
        port = (
            2**15
            + 2**14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
        )
        dist_url = "tcp://127.0.0.1:{}".format(port)
        return dist_url


#! Do we use still want this class???
class DeepDiscEstimator(CatEstimator):
    """DeepDISC estimator"""

    name = "DeepDiscEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(
        cfgfile=Param(str, None, required=True, msg="The primary configuration file for the deepdisc models."),
        batch_size=Param(int, 1, required=False, msg="Batch size of data to load."),
        numclasses=Param(int, 1, required=False, msg="The number of classes in the model."),
        epochs=Param(int, 20, required=False, msg="How many epochs to run estimation."),
        output_dir=Param(str, "./", required=False, msg="The directory to write output to."),
        run_name=Param(str, "run", required=False, msg="Name of the training run."),
        chunk_size=Param(int, 100, required=False, msg="Chunk size used within detectron2 code."),
    )
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
        run_name = self.config.run_name

        cfg = get_lazy_config(cfgfile, batch_size, numclasses)
        cfg_loader = get_loader_config(output_dir, batch_size)

        cfg.train.init_checkpoint = os.path.join(output_dir, run_name) + ".pth"

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
    config_options = {}
    config_options.update(
        cfgfile=Param(str, None, required=True, msg="The primary configuration file for the deepdisc models."),
        batch_size=Param(int, 1, required=False, msg="Batch size of data to load."),
        output_dir=Param(str, "./", required=False, msg="The directory to write output to."),
        run_name=Param(str, "run", required=False, msg="Name of the training run."),
        chunk_size=Param(int, 100, required=False, msg="Chunk size used within detectron2 code."),
        num_camera_filters=Param(int, 6, required=False, msg="The number of camera filters for the dataset used (LSST has 6)."),
    )

    inputs = [("model", ModelHandle),
              ("input", TableHandle),
              ("metadata", JsonHandle)]
    outputs = [("output", QPHandle),
               ("truth", TableHandle)]

    def __init__(self, args, comm=None):
        """Constructor:
        Do Estimator specific initialization"""
        self.nnmodel = None
        CatEstimator.__init__(self, args, comm=comm)

    def estimate(self, input_data, input_metadata):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            self.temp_dir = temp_directory_name
            self.set_data("input", input_data)
            self.set_data("metadata", input_metadata)
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
        self.open_model(**self.config)
        metadata = self.get_data("metadata")

        print("Caching data")
        flattened_image_iterator = self.input_iterator("input")
        for start_idx, _, images in flattened_image_iterator:
            for image_idx, image in enumerate(images["images"]):
                image_metadata = metadata[start_idx + image_idx]
                image_height = image_metadata["height"]
                image_width = image_metadata["width"]

                reformed_image = image.reshape(
                    self.config.num_camera_filters, image_height, image_width
                ).astype(np.float32)

                filename = f"image_{start_idx + image_idx}.npy"
                file_path = os.path.join(self.temp_dir, filename)
                np.save(file_path, reformed_image)
                image_metadata["filename"] = file_path

        cfgfile = self.config.cfgfile
        output_dir = self.config.output_dir

        cfg = LazyConfig.load(cfgfile)
        cfg.OUTPUT_DIR = output_dir

        self.predictor = return_predictor_transformer(cfg, checkpoint=self.nnmodel)

        print("Matching objects")
        true_zs, pdfs = get_matched_z_pdfs(
            metadata,
            DC2ImageReader(),
            lambda dataset_dict: dataset_dict["filename"],
            self.predictor,
        )
        self.true_zs = true_zs
        self.pdfs = np.array(pdfs)

    def finalize(self):
        self.zgrid = np.linspace(0, 5, 200)

        zmode = np.array([self.zgrid[np.argmax(pdf)] for pdf in self.pdfs]).flatten()
        qp_distn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=self.pdfs))
        qp_distn.set_ancil(dict(zmode=zmode))
        qp_distn = self.calculate_point_estimates(qp_distn)
        self.add_handle("output", data=qp_distn)
        truth_dict = dict(redshift=self.true_zs)
        self.add_handle("truth", data=truth_dict)


class DeepDiscPDFEstimatorWithChunking(CatEstimator):
    """DeepDISC estimator"""

    name = "DeepDiscPDFEstimator"
    config_options = {}
    config_options.update(
        cfgfile=Param(str, None, required=True, msg="The primary configuration file for the deepdisc models."),
        batch_size=Param(int, 1, required=False, msg="Batch size of data to load."),
        output_dir=Param(str, "./", required=False, msg="The directory to write output to."),
        run_name=Param(str, "run", required=False, msg="Name of the training run."),
        chunk_size=Param(int, 100, required=False, msg="Chunk size used within detectron2 code."),
        num_camera_filters=Param(int, 6, required=False, msg="The number of camera filters for the dataset used (LSST has 6)."),
        calculated_point_estimates=Param(list, ['mode'], required=False, msg="The point estimates to include by default."),
    )

    inputs = [("model", ModelHandle),
              ("input", TableHandle),
              ("metadata", Hdf5Handle)]
    outputs = [("output", QPHandle),
               ("truth", TableHandle)]

    def __init__(self, args, comm=None):
        """Constructor:
        Do Estimator specific initialization"""
        CatEstimator.__init__(self, args, comm=comm)

        self.nnmodel = None
        self.zgrid = np.linspace(0, 5, 200)
        self._output_handle = None

    def estimate(self, input_data, input_metadata):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            self.temp_dir = temp_directory_name
            self.set_data("input", input_data)
            self.set_data("metadata", input_metadata)
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
        self.open_model(**self.config)

        flattened_image_iterator = self.input_iterator("input")
        metadata_iterator = self.input_iterator("metadata")
        is_first = True

        print("Caching data")
        for start_idx, end_idx, images, _, _, metadata_json_dicts in zip(flattened_image_iterator, metadata_iterator):

            # Convert the json into dicts and load them into a list
            metadata = [json.loads(this_json) for this_json in metadata_json_dicts['metadata_dicts']]

            # Reform the flattened image, update metadata with cached image file path
            for image_idx, image in enumerate(images["images"]):
                image_metadata = metadata[image_idx]
                image_height = image_metadata["height"]
                image_width = image_metadata["width"]

                reformed_image = image.reshape(
                    self.config.num_camera_filters, image_height, image_width
                ).astype(np.float32)

                filename = f"image_{start_idx + image_idx}.npy"
                file_path = os.path.join(self.temp_dir, filename)
                np.save(file_path, reformed_image)
                image_metadata["filename"] = file_path

            # process this chunk of data
            print(f"Processing chunk (start:end) - ({start_idx}:{end_idx})")
            self._process_chunk(start_idx, end_idx, metadata, is_first)
            is_first = False

        # close the output file.
        print("Finalizing the output file")
        self._finalize_run()

    def _process_chunk(self, start_idx, end_idx, metadata, is_first):
        cfg = LazyConfig.load(self.config.cfgfile)
        cfg.OUTPUT_DIR = self.config.output_dir

        self.predictor = return_predictor_transformer(cfg, checkpoint=self.nnmodel)

        print("Matching objects")
        true_zs, pdfs = get_matched_z_pdfs(
            metadata,
            DC2ImageReader(),
            lambda dataset_dict: dataset_dict["filename"],
            self.predictor,
        )
        self.true_zs = true_zs
        self.pdfs = np.array(pdfs)

        # add this chunk of pdfs to a qp.ensemble
        print("Adding PDFs to ensemble")
        qp_ensemble = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=self.pdfs))
        print("Adding true Z to ensemble")
        qp_ensemble.set_ancil(dict(true_zs=self.true_zs))

        # write the partial ensemble to a file
        print("Writing out this ensemble chunk to disk")
        self._do_chunk_output(qp_ensemble, start_idx, end_idx, is_first)
