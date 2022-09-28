import argparse
import logging
import multiprocessing
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from copy import deepcopy
from datetime import datetime, timedelta
from multiprocessing import Process
from typing import Tuple, List

import torch
import torchvision

from torch_stress.device_monitor import nvml_device_count


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


def get_correct_outputs(cpu_model: torch.nn.Module, inputs: torch.Tensor) -> torch.tensor:
    return cpu_model(inputs)


def stress_by_training(device: torch.device, model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
                       correct_outputs: torch.Tensor, queue: multiprocessing.Queue, stop_time: datetime):
    inputs_device = inputs.to(device)
    labels_device = labels.to(device)
    while True:
        outputs = model(inputs_device)
        loss = torch.nn.functional.cross_entropy(outputs, labels_device)
        model.zero_grad()
        loss.backward()
        outputs_cpu = outputs.to('cpu')
        # count errors instead of asserting
        errors = (outputs_cpu != correct_outputs).sum().item()
        queue.put((device, errors))
        if datetime.now() > stop_time:
            queue.put((datetime.now(), f'Runtime limit for device {device} reached - terminating.'))
            break


def stress(device: torch.device, queue: multiprocessing.Queue, stop_time: datetime):
    configure()
    model_cpu, input_size, num_classes = get_model()
    model_device = deepcopy(model_cpu).to(device)
    max_bs = determine_max_batch_size(device, model_cpu, input_size, num_classes)
    queue.put((datetime.now(), f'Selected batch size {max_bs} for device {device}.'))
    exemplary_inputs, labels = generate_inputs(max_bs, input_size, num_classes)
    correct_outputs = get_correct_outputs(model_cpu, exemplary_inputs)
    stress_by_training(device, model_device, exemplary_inputs, labels, correct_outputs, queue, stop_time)


def spawn_stress_processes(args: argparse.Namespace) -> Tuple[List[multiprocessing.Process], multiprocessing.Queue]:
    multiprocessing.set_start_method('spawn')
    queue = multiprocessing.Queue()
    num_devices = nvml_device_count()
    logging.info('Spawning the stress processes.')
    now = datetime.now()
    stop_time = now + timedelta(seconds=args.runtime)
    processes = []
    for device in range(num_devices):
        p = Process(target=stress, args=(device, queue, stop_time))
        processes.append(p)
    for p in processes:
        p.start()
    return processes, queue


def stop_stress_processes(processes: List[multiprocessing.Process]):
    for p in processes:
        # TODO replace with something safe
        # see https://docs.python.org/3/library/multiprocessing.html#programming-guidelines
        p.terminate()
