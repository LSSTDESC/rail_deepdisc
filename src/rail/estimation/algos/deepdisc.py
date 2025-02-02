import json
import os
import sys
import tempfile
from collections import namedtuple
from fvcore.common.checkpoint import Checkpointer
import detectron2.data as d2data
import detectron2.solver as solver
import detectron2.utils.comm as comm
import numpy as np
import qp
from ceci.config import StageParameter as Param
from ceci.stage import PipelineStage
from deepdisc.data_format.augment_image import dc2_train_augs
from deepdisc.data_format.image_readers import DC2ImageReader
from deepdisc.inference.match_objects import (run_batched_match_redshift,
                                              run_batched_get_object_coords,
                                              get_matched_object_classes_new,
                                              get_matched_z_pdfs,
                                              get_matched_z_pdfs_new,
                                              get_matched_z_points_new)
from deepdisc.inference.predictors import return_predictor_transformer
from deepdisc.model.loaders import return_test_loader, return_train_loader
from deepdisc.model.models import return_lazy_model
from deepdisc.training.trainers import (return_evallosshook,
                                        return_lazy_trainer, return_optimizer,
                                        return_savehook, return_schedulerhook)
from detectron2.config import LazyConfig, get_cfg, instantiate, CfgNode
from detectron2.engine import launch
from detectron2.engine.defaults import create_ddp_model
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import Hdf5Handle, ModelHandle, QPHandle, TableHandle
from rail.estimation.estimator import CatEstimator, CatInformer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue as TorchQueue

# temp file namedtuple for start_idx, file_name, total number of pdfs, and file handle
TempFileMeta = namedtuple('TempFileMeta', ['start_idx', 'file_name', 'total_pdfs', 'file_handle'])


def train(config, all_metadata, train_head=True):
    
    cfgfile = config["cfgfile"]
    batch_size = config["batch_size"]
    output_dir = config["output_dir"]
    run_name = config["run_name"]
    print_frequency = config["print_frequency"]
    epoch = config["epoch"]
    head_epochs = config["head_epochs"]
    full_epochs = config["full_epochs"]
    mile1 = config["mile1"]
    mile2 = config["mile2"]
    training_percent = config["training_percent"]

    e1 = epoch * head_epochs
    e2 = epoch * mile1
    e3 = epoch * mile2
    #efinal = epoch * full_epochs
    
    efinal = int(epoch * (full_epochs+head_epochs))

    
    val_per = epoch

    # Create slices for the input data
    total_images = len(all_metadata)
    split_index = int(np.floor(total_images * training_percent))
    train_slice = slice(split_index)
    eval_slice = slice(split_index, total_images)

    
    cfg = LazyConfig.load(cfgfile)
    cfg.OUTPUT_DIR = output_dir
    
    mapper = cfg.dataloader.train.mapper(
        cfg.dataloader.imagereader, lambda dataset_dict: dataset_dict["filename"], cfg.dataloader.augs
    ).map_data

    training_loader = d2data.build_detection_train_loader(
        all_metadata[train_slice], mapper=mapper, total_batch_size=batch_size
    )
    

    
    model = return_lazy_model(cfg, freeze=False)

    saveHook = return_savehook(run_name, epoch)


    if train_head:
        
        cfg.optimizer.params.model = model
        cfg.SOLVER.MAX_ITER = e1  # for DefaultTrainer
        
        optimizer = solver.build_optimizer(cfg, model)
        schedulerHook = return_schedulerhook(optimizer)
        
        if training_percent >= 1.0:
            # don't do lossHook
            hookList = [schedulerHook, saveHook]
            print(f"The validation loss has been omitted, as the training percent is {training_percent}. To include it, set the training percent to a value between 0 and 1.")
        else:
            eval_loader = d2data.build_detection_test_loader(all_metadata[eval_slice], mapper=mapper, batch_size=batch_size)
            lossHook = return_evallosshook(val_per, model, eval_loader)
            hookList = [lossHook, schedulerHook, saveHook]


        trainer = return_lazy_trainer(
            model, training_loader, optimizer, cfg, hookList
        )

        trainer.set_period(print_frequency)

        trainer.train(0, e1)

        if comm.is_main_process():
            np.save(os.path.join(output_dir,run_name) + "_losses.npy", trainer.lossList)
            if training_percent<1.0:
                np.save(output_dir + run_name + "_val_losses", trainer.vallossList)

    else:
        cfg.train.init_checkpoint = os.path.join(output_dir, run_name + ".pth")
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.MAX_ITER = efinal  # for DefaultTrainer
        cfg.SOLVER.STEPS=[e2,e3]
        
        cfg.optimizer.lr = 0.0001
        
        optimizer = solver.build_optimizer(cfg, model)
        schedulerHook = return_schedulerhook(optimizer)

        if training_percent >= 1.0:
            # don't do lossHook
            hookList = [schedulerHook, saveHook]
            print(f"The validation loss has been omitted, as the training percent is {training_percent}. To include it, set the training percent to a value between 0 and 1.")
        else:
            eval_loader = d2data.build_detection_test_loader(all_metadata[eval_slice], mapper=mapper, batch_size=batch_size)
            lossHook = return_evallosshook(val_per, model, eval_loader)
            hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(
            model, training_loader, optimizer, cfg, hookList
        )

        trainer.set_period(print_frequency)
        trainer.train(e1, efinal)

        if comm.is_main_process():
            losses = np.load(os.path.join(output_dir,run_name) + "_losses.npy")
            losses = np.concatenate((losses, trainer.lossList))
            np.save(os.path.join(output_dir,run_name) + "_losses.npy", losses)
            if training_percent<1.0:
                vallosses = np.load(output_dir + run_name + "_val_losses.npy")
                vallosses = np.concatenate((vallosses, trainer.vallossList))
                np.save(output_dir + run_name + "_val_losses", vallosses)


class DeepDiscInformer(CatInformer):
    """This informer can parallelize model training with input data across
    multiple GPUs.

    The stage configuration parameter `batch_size` defines the number of images
    to send to a GPU at a time while training the model.
    """

    name = "DeepDiscInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(
        cfgfile=Param(str, None, required=True, msg="The primary configuration file for the deepdisc models."),

        batch_size=Param(int, 1, required=False, msg="Number of images sent to each GPU per node for parallel training."),
        epoch=Param(int, 20, required=False, msg="Number of iterations per epooch."),
        full_epochs=Param(int, 0, required=False, msg="How many iterations when training the head layers and unfrozen backbone layers together."),
        mile1=Param(int, 0, required=False, msg="Milestone 1 for param scheduler.  Number of epochs"),
        mile2=Param(int, 0, required=False, msg="Milestone 2 for param scheduler.  Number of epochs"),        
        head_epochs=Param(int, 0, required=False, msg="How many iterations when training the head layers (while the backbone layers are frozen)."),
        machine_rank=Param(int, 0, required=False, msg="The rank of this machine."),
        num_camera_filters=Param(int, 6, required=False, msg="The number of camera filters for the dataset used (LSST has 6)."),
        num_gpus=Param(int, 1, required=False, msg="Number of processes per machine. When using GPUs, this should be the number of GPUs."),
        num_machines=Param(int, 1, required=False, msg="The total number of machines."),
        output_dir=Param(str, "./", required=False, msg="The directory to write output to."),
        print_frequency=Param(int, 5, required=False, msg="How often to print in-progress output (happens every x number of iterations)."),
        run_name=Param(str, "run", required=False, msg="Name of the training run."),
        training_percent=Param(float, 0.8, required=False, msg="The fraction of input data used to split into training/evaluation sets."),
    )
    inputs = [('input', TableHandle), ('metadata', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # check to make sure that batch_size is an even multiple of num_gpus
        if self.config.batch_size % self.config.num_gpus != 0:
            raise ValueError(f"batch_size ({self.config.batch_size}) must be an even multiple of num_gpus ({self.config.num_gpus})")

    def inform(self, input_data, input_metadata):
        with tempfile.TemporaryDirectory() as temp_directory_name:
            self.temp_dir = temp_directory_name
            self.set_data("input", input_data, do_read=False)
            self.set_data("metadata", input_metadata,do_read=False)
            self.run()
            self.finalize()
        return self.get_handle("model")

    def run(self):
        """
        Train a inception NN on a fraction of the training data
        """
        self.metadata = []

        print("Caching data")

        # create iterators for both the flattened images and the metadata
        flattened_image_iterator = self.input_iterator("input")
        metadata_iterator = self.input_iterator("metadata")

        # iterate over the flattened images and metadata in parallel
        for image_chunk, json_chunk in zip(flattened_image_iterator, metadata_iterator):
            start_idx, _, images = image_chunk
            _, _, metadata_json_dicts = json_chunk

            # convert the json into dicts and load them into a list
            metadata_chunk = [json.loads(this_json) for this_json in metadata_json_dicts['metadata_dicts']]

            # reform the flattened image, update metadata with cached image file path
            for image_idx, image in enumerate(images["images"]):
                this_image_metadata = metadata_chunk[image_idx]
                image_height = this_image_metadata["height"]
                image_width = this_image_metadata["width"]

                reformed_image = image.reshape(
                    self.config.num_camera_filters, image_height, image_width
                ).astype(np.float32)

                filename = f"image_{start_idx + image_idx}.npy"
                file_path = os.path.join(self.temp_dir, filename)
                np.save(file_path, reformed_image)

                this_image_metadata["filename"] = file_path

            # add this chunk of metadata to the list of metadata
            self.metadata.extend(metadata_chunk)

        dist_url = _get_dist_url()

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

        #print("Training full model")
        print("Training head layers")
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

def _get_dist_url():
    port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )
    dist_url = "tcp://127.0.0.1:{}".format(port)
    return dist_url

def _do_inference(q, cfg, predictor, metadata, num_gpus, batch_size, zgrid, dist_url):

        """This is the function that is called by `launch` to parallelize
        inference across all available GPUs."""

        #group = dist.new_group()
        group=None
        
        
        mapper = cfg.dataloader.test.mapper(
            DC2ImageReader(), lambda dataset_dict: dataset_dict["filename"],
        ).map_data

        loader = d2data.build_detection_test_loader(
            metadata, mapper=mapper, batch_size=batch_size
        )

        # this batched version will break up the metadata across GPUs under the hood.
        #true_zs, pdfs, ids, blendedness = run_batched_match_redshift(loader, predictor, ids=True, blendedness=True)
        pdfs, ras, decs, classes, gmms, scores = run_batched_get_object_coords(loader, predictor, gmm=True)

        # convert the python lists into numpy arrays
        pdfs = np.array(pdfs)
        ras = np.array(ras)
        decs = np.array(decs)
        classes = np.array(classes)
        gmms = np.array(gmms)
        scores = np.array(scores)


        if dist.get_rank() == 0:
            # Create temporary lists to hold the pdfs, true_zs, and ids from each process
            pdfs_list = [None for _ in range(num_gpus)]
            ras_list = [None for _ in range(num_gpus)]
            decs_list = [None for _ in range(num_gpus)]
            classes_list = [None for _ in range(num_gpus)]
            gmms_list = [None for _ in range(num_gpus)]
            scores_list = [None for _ in range(num_gpus)]


            # gather the pdfs, true_zs, and ids from all the processes.
            dist.gather_object(pdfs, object_gather_list=pdfs_list, dst=0, group=group)
            dist.gather_object(ras, object_gather_list=ras_list, dst=0, group=group)
            dist.gather_object(decs, object_gather_list=decs_list, dst=0, group=group)
            dist.gather_object(classes, object_gather_list=classes_list, dst=0, group=group)
            dist.gather_object(gmms, object_gather_list=gmms_list, dst=0, group=group)
            dist.gather_object(scores, object_gather_list=scores_list, dst=0, group=group)

            # concatenate all the gathered outputs so they can be added to a qp.ensemble.
            all_pdfs = np.concatenate(pdfs_list)
            all_ras = np.concatenate(ras_list)
            all_decs = np.concatenate(decs_list)
            all_classes = np.concatenate(classes_list)
            all_gmms = np.concatenate(gmms_list)
            all_scores = np.concatenate(scores_list)

            if len(all_pdfs):
                # Add all the pdfs and ancil data to a qp.ensemble
                qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=all_pdfs))
                qp_dstn.set_ancil(dict(ra=all_ras))
                qp_dstn.add_to_ancil(dict(dec=all_decs))
                qp_dstn.add_to_ancil(dict(oclass=all_classes))
                qp_dstn.add_to_ancil(dict(gmm=all_gmms))
                qp_dstn.add_to_ancil(dict(scores=all_scores))

                # add the qp Ensemble to the queue so it can be picked up and written to disk.
                q.put(qp_dstn)
            else:
                # no pdfs found, adding `None` to the queue
                q.put(None)

        else:
            dist.gather_object(pdfs, object_gather_list=None, dst=0, group=group)
            dist.gather_object(ras, object_gather_list=None, dst=0, group=group)
            dist.gather_object(decs, object_gather_list=None, dst=0, group=group)
            dist.gather_object(classes, object_gather_list=None, dst=0, group=group)
            dist.gather_object(gmms, object_gather_list=None, dst=0, group=group)
            dist.gather_object(scores, object_gather_list=None, dst=0, group=group)


class DeepDiscPDFEstimatorWithChunking(CatEstimator):
    """This estimator can distribute and parallelize processing of input data both
    horizontally across nodes and vertically across GPUs on each node.

    Initially this stage will break up the input data into set with size `chunk_size`.
    Those data sets will be distributed across the available compute nodes.

    Each compute node will have 1 or more associated GPUs to run inference on the
    data. The data set on the node will be processed in parallel across the available
    GPUs in subsets of size `batch_size`.

    Because of the way the input data is distributed and then parallelized, generally
    `chunk_size` >= `batch_size`.

    The results of inference across all GPUs and nodes will be recombined and written
    out to a single output file."""

    name = "DeepDiscPDFEstimatorWithChunking"
    config_options = {}
    config_options.update(
        cfgfile=Param(str, None, required=True, msg="The primary configuration file for the deepdisc models."),
        batch_size=Param(int, 1, required=False, msg="Number of images sent to each GPU per node for parallel processing."),
        calculated_point_estimates=Param(list, ['mode'], required=False, msg="The point estimates to include by default."),
        chunk_size=Param(int, 100, required=False, msg="Number of images distributed to each node for processing."),
        num_gpus=Param(int, 2, required=False, msg="Number of processes per machine. When using GPUs, this should be the number of GPUs per machine."),
        num_camera_filters=Param(int, 6, required=False, msg="The number of camera filters for the dataset used (LSST has 6)."),
        output_dir=Param(str, "./", required=False, msg="The directory to write output to."),
        #return_ids_with_inference=Param(bool, False, required=False, msg="Whether to return the ids with the results of inference."),
        #return_bnds_with_inference=Param(bool, False, required=False, msg="Whether to return the object blendedness with the results of inference."),
        run_name=Param(str, "run", required=False, msg="Name of the training run."),
    )

    inputs = [("model", ModelHandle),
              ("input", TableHandle),
              ("metadata", Hdf5Handle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args, **kwargs):
        """Constructor:
        Do Estimator specific initialization"""
        super().__init__(args, **kwargs)

        self.nnmodel = None
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        self._output_handle = None
        self._temp_file_meta_tuples = []

    def estimate(self, input_data, input_metadata):
        print('setting data')
        self.set_data("input", input_data, do_read=False)
        self.set_data("metadata", input_metadata, do_read=False)
        with tempfile.TemporaryDirectory() as temp_directory_name:
            self.temp_dir = temp_directory_name
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
        # Keep this here!!! (necessary for some reason)
        # Only needs to be set one time
        #mp.set_start_method('spawn')

        self.open_model(**self.config)
        
        cfg = LazyConfig.load(self.config.cfgfile)
        cfg.OUTPUT_DIR = self.config.output_dir
        
        
        self.predictor = return_predictor_transformer(cfg, checkpoint=self.nnmodel)
        flattened_image_iterator = self.input_iterator("input")
        metadata_iterator = self.input_iterator("metadata")
        
        if self.config.num_gpus==1:
            dist.init_process_group(
            backend="NCCL",
            init_method=_get_dist_url(),
            world_size=1,
            rank=0,
        )

        print("Caching data")

        for image_chunk, json_chunk in zip(flattened_image_iterator, metadata_iterator):
            start_idx, end_idx, images = image_chunk
            _, _, metadata_json_dicts = json_chunk

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
            self._process_chunk(start_idx, metadata)
            

    def _process_chunk(self, start_idx, metadata):
        """For a given block of images and metadata, calculate the PDFs and
        write them to a temporary file.

        Parameters
        ----------
        start_idx : int
            The starting index of the block of images
        metadata : list[dict]
            The list of metadata dictionaries for this block of images
        """

        cfg = LazyConfig.load(self.config.cfgfile)

        
        with mp.Manager() as manager:
            q = manager.Queue()

            # call detectron2's `launch` function to parallelize the inference
            launch(
                _do_inference,
                num_gpus_per_machine=self.config.num_gpus,
                # num_machines=1 ??? I don't think we need this
                # machine_rank=self.rank ??? I don't think we need this, I could be wrong
                dist_url=_get_dist_url(),
                args=(
                    q,
                    cfg,
                    self.predictor,
                    metadata,
                    self.config.num_gpus,
                    self.config.batch_size,
                    self.zgrid,
                    #self.config.return_ids_with_inference,
                    #self.config.return_bnds_with_inference,
                    _get_dist_url(),
                ),
            )

            # Check the queue and grab the qp.ensemble if it's there
            if q.qsize():
                qp_dstn = q.get()

                # if there are pdfs in the qp.ensemble, calculate point estimates and
                # write the qp.ensemble to a temporary file.
                if qp_dstn is not None and qp_dstn.npdf:
                    qp_dstn = self.calculate_point_estimates(qp_dstn)
                    temp_file_tuple = self._write_temp_file(qp_dstn, start_idx)
                    self._temp_file_meta_tuples.append(temp_file_tuple)



    def _write_temp_file(self, qp_dstn, start_idx):
        """Write the qp.ensemble to a temporary file and return a named tuple
        that contains the start index, file name, total number of pdfs, and data
        handle for the temporary file.

        Parameters
        ----------
        qp_dstn : qp.Ensemble
            The qp.ensemble to write to the temporary file
        start_idx : int
            The starting index of the block of images
        """

        num_pdfs = qp_dstn.npdf

        if num_pdfs == 0:
            return TempFileMeta(start_idx, None, 0, None)
        else:
            # create the temporary file path
            file_path = os.path.join(self.temp_dir, f"pdfs_{start_idx}.hdf5")

            # write the qp.ensemble to the temporary file
            tmp_handle = QPHandle(tag=file_path, path=file_path, data=qp_dstn)
            tmp_handle.initialize_write(data_length=num_pdfs, communicator=self.comm)
            tmp_handle.write()
            tmp_handle.finalize_write()

            # use a TempFileMeta tuple to track this temporary file
            return TempFileMeta(start_idx, file_path, num_pdfs, tmp_handle)

    def _do_chunk_output(self, qp_dstn, start, end, first):
        """Function that adds a qp Ensemble to a given output file.

        Parameters
        ----------
        qp_dstn : qp.Ensemble
            The qp.ensemble to write to the output file
        start : int
            The starting index to write the qp.ensemble to
        end : int
            The ending index to write the qp.ensemble to
        first : bool
            Whether this is the first time writing to the output file, used to
            initialize the file.
        """
        if first:
            self._output_handle = self.add_handle('output', data=qp_dstn)
            self._output_handle.initialize_write(self.total_pdfs, communicator=self.comm)
        self._output_handle.set_data(qp_dstn, partial=True)
        self._output_handle.write_chunk(start, end)

    def finalize(self):
        """Creates the final output file.
        Sort the list of temporary files using the start_idx and then write the
        contents of each temporary file to the final output file.
        """

        # if no temporary files were created, create an empty output file and return
        if len(self._temp_file_meta_tuples) == 0:
            self.add_handle('output', data=None)
            return

        # sort self._temp_file_meta_tuples by start_idx
        self._temp_file_meta_tuples.sort(key=lambda x: x.start_idx)

        # find the total number of output PDFs for all the temporary files
        self.total_pdfs=0
        for meta in self._temp_file_meta_tuples:
            self.total_pdfs += meta.total_pdfs

        # open each temporary file, write it's contents to the final file.
        is_first = True
        previous_index = 0
        for meta in self._temp_file_meta_tuples:
            tmp_handle = meta.file_handle
            qp_dstn = tmp_handle.read()
            self._do_chunk_output(qp_dstn, previous_index, previous_index + meta.total_pdfs, is_first)
            previous_index += meta.total_pdfs
            is_first = False

        # finalize the output file
        self._output_handle.finalize_write()

        # call the super class to finalize any parallelization work happening
        PipelineStage.finalize(self)
