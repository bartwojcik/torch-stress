import logging

from pynvml import *

_nvml_initialized = False
_device_count = None
_device_handles = []


def nvml_init():
    global _nvml_initialized, _device_count, _device_handles
    if _nvml_initialized:
        return
    nvmlInit()
    _device_count = nvmlDeviceGetCount()
    for i in range(_device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        _device_handles.append(handle)
    return _device_count


def nvml_device_count():
    return _device_count


def nvml_get_serial(device: int):
    handle = _device_handles[device]
    return nvmlDeviceGetSerial(handle)


def nvml_get_name(device: int):
    handle = _device_handles[device]
    return nvmlDeviceGetName(handle)


def nvml_get_name_and_serial(device: int):
    return f'{nvml_get_name(device)} ({nvml_get_serial(device)})'


def nvml_get_temp(device: int):
    handle = _device_handles[device]
    results = {}
    try:
        results['gpu_temp'] = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    except NVMLError as e:
        logging.error(f'nvmlDeviceGetTemperature: {e}')
    return results


def nvml_get_temp_thresholds(device: int):
    handle = _device_handles[device]
    thresholds = {
        'temp_slowdown': NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
        'temp_shutdown': NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
        'temp_max_gpu': NVML_TEMPERATURE_THRESHOLD_GPU_MAX,
        'temp_max_mem': NVML_TEMPERATURE_THRESHOLD_MEM_MAX,
    }
    results = {}
    for name, enum in thresholds.items():
        try:
            thr = nvmlDeviceGetTemperatureThreshold(handle, enum)
            results[name] = thr
        except NVMLError as e:
            logging.error(f'{enum}: {e}')
    return results


def nvml_get_fanspeed(device: int):
    handle = _device_handles[device]
    results = {}
    try:
        num_fans = nvmlDeviceGetNumFans(handle)
    except NVMLError as e:
        logging.error(f'nvmlDeviceGetNumFans: {e}')
        return results
    for i in range(num_fans):
        try:
            fan_speed = nvmlDeviceGetFanSpeed_v2(handle, i)
            results[i] = fan_speed
        except NVMLError as e:
            logging.error(f'nvmlDeviceGetFanSpeed_v2 (fan {i}): {e}')
    return results


# TODO add throttling reasons info

def nvml_get_utilization(device: int):
    handle = _device_handles[device]
    results = {}
    try:
        util_rates = nvmlDeviceGetUtilizationRates(handle)
        results['gpu_util_rate'] = util_rates.gpu
        results['mem_util_rate'] = util_rates.memory
    except NVMLError as e:
        logging.error(f'nvmlDeviceGetUtilizationRates: {e}')
    return results


def nvml_get_mem_usage(device: int):
    handle = _device_handles[device]
    results = {}
    try:
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        results['mem_used'] = mem_info.used
        results['mem_total'] = mem_info.total
    except NVMLError as e:
        logging.error(f'nvmlDeviceGetMemoryInfo: {e}')
    return results


def nvml_shutdown():
    global _nvml_initialized
    if _nvml_initialized:
        nvmlShutdown()
