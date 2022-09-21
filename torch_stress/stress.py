import argparse
import logging
import multiprocessing
from copy import deepcopy
from multiprocessing import Process
from time import sleep
from typing import Tuple

import torch
import torchvision


def configure():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model() -> Tuple[torch.nn.Module, Tuple, int]:
    # TODO add an argument for selecting a model
    # TODO allow overriding input size for models that accept inputs with arbitrary size
    model = torchvision.models.vit_b_16()
    input_size = (3, model.image_size, model.image_size)
    num_classes = model.num_classes
    return model, input_size, num_classes


def determine_max_batch_size(device: torch.device, model: torch.nn.Module, input_size: Tuple[int],
                             num_classes: int) -> int:
    current_bs = 2
    while True:
        try:
            inputs = torch.randn(current_bs, *input_size, device=device)
            labels = torch.randint(low=0, high=num_classes, size=(current_bs,), device=device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()
            current_bs *= 2
        except RuntimeError:
            return current_bs // 2


def generate_inputs(batch_size: int, input_size: Tuple[int], num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randn(batch_size, *input_size)
    labels = torch.randint(low=0, high=num_classes, size=(batch_size,))
    return inputs, labels


def get_correct_outputs(cpu_model: torch.nn.Module, inputs: torch.Tensor):
    return cpu_model(inputs)


def stress_by_training(device: torch.device, model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
                       correct_outputs: torch.Tensor):
    inputs_device = inputs.to(device)
    labels_device = labels.to(device)
    while True:
        outputs = model(inputs_device)
        loss = torch.nn.functional.cross_entropy(outputs, labels_device)
        model.zero_grad()
        loss.backward()
        outputs_cpu = outputs.to('cpu')
        # count errors instead of asserting
        assert torch.equal(outputs_cpu, correct_outputs)
        logging.info(f'Device {device} loop OK.')


def stress(args: argparse.Namespace, device: torch.device):
    configure()
    model_cpu, input_size, num_classes = get_model()
    model_device = deepcopy(model_cpu).to(device)
    max_bs = determine_max_batch_size(device, model_cpu, input_size, num_classes)
    logging.info(f'Selected batch size {max_bs} for device {device}.')
    exemplary_inputs, labels = generate_inputs(max_bs, input_size, num_classes)
    correct_outputs = get_correct_outputs(model_cpu, exemplary_inputs)
    stress_by_training(device, model_device, exemplary_inputs, labels, correct_outputs)


def stress_processes_spawn(args: argparse.Namespace):
    multiprocessing.set_start_method('spawn')
    logging.info('Spawning the stress processes.')
    # TODO enumerate and get all gpu devices, one process for each device
    # only one process for now
    p = Process(target=stress, args=(args, 0))
    p.start()
    # TODO print temperatures and device utilization instead, interactively (what about logging?)
    # TODO additionally, verify whether all processes are not stuck and progress with their loops
    sleep(args.runtime)
    logging.info('Runtime length reached - terminating.')
    p.terminate()
