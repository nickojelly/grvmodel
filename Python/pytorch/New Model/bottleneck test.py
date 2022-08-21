import torch
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import PyTorchProfiler
from datamodule import BoringDataModule
from model import BoringModel

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("lightning_logs")
)
trainer = Trainer(
    profiler=profiler
)