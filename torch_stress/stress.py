import argparse
import logging
import multiprocessing
import os
from queue import Empty

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from datetime import datetime, timedelta
from multiprocessing import Process
from typing import Tuple, List

import torch
import torchvision

from torch_stress.device_monitor import nvml_device_count


def configure():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # sys.stdout.close()
    # sys.stderr.close()


def get_model() -> Tuple[torch.nn.Module, Tuple, int]:
    # TODO add an argument for selecting a model
    # TODO allow overriding input size for models that accept inputs with arbitrary size?
    model = torchvision.models.vit_b_16(weights='DEFAULT', progress=False)
    model.eval()
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


def stress_by_training(process_ind: int, device_ind: int, model: torch.nn.Module, inputs: torch.Tensor,
                       labels: torch.Tensor,
                       cpu_outputs: torch.Tensor, status_queue: multiprocessing.Queue,
                       stop_queue: multiprocessing.Queue,
                       stop_time: datetime):
    device = torch.device(f'cuda:{device_ind}')
    device_inputs = inputs.to(device)
    device_labels = labels.to(device)
    saved_outputs = model(device_inputs).detach().cpu()
    # CPU to device numerical precision test
    # TODO possibly move the allowed errors and pair them with model
    errors = torch.isclose(cpu_outputs, saved_outputs, atol=5e-4, rtol=0.0).logical_not().sum().item()
    max_abs_error = (cpu_outputs - saved_outputs).abs().max()
    status_queue.put((device_ind, errors))
    status_queue.put((datetime.now(),
                      f'Got {errors} errors from the CPU vs device {device_ind} test. Max absolute error is: {max_abs_error}'))
    del cpu_outputs
    # device stability device test
    while True:
        outputs = model(device_inputs)
        loss = torch.nn.functional.cross_entropy(outputs, device_labels)
        model.zero_grad()
        loss.backward()
        outputs = outputs.detach().cpu()
        errors = (outputs != saved_outputs).sum().item()
        status_queue.put((device_ind, errors))
        if datetime.now() > stop_time:
            status_queue.put((datetime.now(),
                              f'Runtime limit for process {process_ind} (device {device_ind}) reached - terminating.'))
            break
        else:
            try:
                stop_queue.get_nowait()
                status_queue.put((datetime.now(),
                                  f'Stop signal received for process {process_ind} (device {device_ind}) - terminating.'))
                break
            except Empty:
                pass


def stress(process_ind: int, device_ind: int, cpu_model: torch.nn.Module,
           status_queue: multiprocessing.Queue, stop_queue: multiprocessing.Queue, stop_time: datetime):
    configure()
    model, input_size, num_classes = get_model()
    device = torch.device(f'cuda:{device_ind}')
    device_model = model.to(device)
    max_bs = determine_max_batch_size(device, device_model, input_size, num_classes)
    status_queue.put((datetime.now(), f'Selected batch size {max_bs} for process {process_ind} (device {device_ind}).'))
    exemplary_inputs, labels = generate_inputs(max_bs, input_size, num_classes)
    cpu_outputs = cpu_model(exemplary_inputs)
    stress_by_training(process_ind, device_ind, device_model, exemplary_inputs, labels, cpu_outputs, status_queue,
                       stop_queue, stop_time)


def spawn_stress_processes(args: argparse.Namespace) -> Tuple[List[multiprocessing.Process], multiprocessing.Queue]:
    multiprocessing.set_start_method('spawn')
    status_queue = multiprocessing.Queue()
    stop_queue = multiprocessing.Queue()
    num_devices = nvml_device_count()
    logging.debug('Instantiating the model.')
    cpu_model, input_size, num_classes = get_model()
    cpu_model.share_memory()
    exemplary_input, labels = generate_inputs(1, input_size, num_classes)
    assert (cpu_model(exemplary_input).detach() != cpu_model(
        exemplary_input).detach()).sum().item() == 0, f'Even the CPU model is nondeterministic!'
    logging.info('Starting the stress processes.')
    now = datetime.now()
    stop_time = now + timedelta(seconds=args.runtime)
    processes = []
    process_ind = 0
    for device_ind in range(num_devices):
        p = Process(target=stress, args=(process_ind, device_ind, cpu_model, status_queue, stop_queue, stop_time))
        processes.append(p)
        process_ind += 1
    for p in processes:
        p.start()
    return processes, status_queue, stop_queue


def terminate_stress_processes(processes: List[multiprocessing.Process], stop_queue: multiprocessing.Queue):
    for _ in range(len(processes)):
        stop_queue.put(None)
    for p in processes:
        p.join(5.0)
        if p.exitcode is None:
            p.terminate()
