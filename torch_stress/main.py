import argparse
import logging
import os
from datetime import datetime

import urwid

from torch_stress.device_monitor import nvml_shutdown, nvml_init, nvml_device_count, nvml_get_name, \
    nvml_get_temp_thresholds, nvml_get_temp, nvml_get_fanspeed, nvml_get_utilization, nvml_get_mem_usage

# key, font color, background color
palette = [
    ('devices_title', 'dark magenta,bold', ''),
    ('devices', 'light magenta,bold', ''),
    ('logs_title', 'yellow', ''),
    ('logs', 'white', ''),
]


class DeviceInfoBox(urwid.WidgetWrap):
    def __init__(self):
        self.devices_text = urwid.Text('Loading...')
        self.devices_body = urwid.AttrMap(
            urwid.LineBox(urwid.Filler(urwid.AttrMap(self.devices_text, 'devices')),
                          title='Devices',
                          title_align='left'),
            'devices_title'
        )
        super().__init__(self.devices_body)
        self.initial_read()
        self.refresh()

    def initial_read(self):
        self._num_devices = nvml_device_count()
        self.ds = {}
        for device in range(self._num_devices):
            self.ds[device] = {}
            # self.ds[device]['serial'] = nvml_get_serial(device)
            self.ds[device]['name'] = nvml_get_name(device)
            self.ds[device]['thresholds'] = nvml_get_temp_thresholds(device)
        self.update_readings()

    def update_readings(self):
        self.now = datetime.now()
        for device in range(self._num_devices):
            self.ds[device]['temp'] = nvml_get_temp(device)
            # self.ds[device]['fans'] = nvml_get_fanspeed(device)
            self.ds[device]['util'] = nvml_get_utilization(device)
            self.ds[device]['mem'] = nvml_get_mem_usage(device)

    def refresh(self):
        self.update_readings()
        # TODO add number of compute errors
        # TODO add time since last update from the process, remove current time
        device_texts = [f'{self.now.strftime("%Y-%m-%d %H:%M:%S")}',
                        'name | temp / slowdown temp | '
                        'gpu utilization | mem utilization | mem used / mem total']
        for device in range(self._num_devices):
            # f"{self.ds[device]['name']} ({self.ds[device]['serial']}) | " \
            device_text = f"{self.ds[device]['name']} | " \
                          f"{self.ds[device]['temp']['gpu_temp']}C / " \
                          f"{self.ds[device]['thresholds']['temp_slowdown']}C | " \
                          f"{self.ds[device]['util']['gpu_util_rate']}% | " \
                          f"{self.ds[device]['util']['mem_util_rate']}% | " \
                          f"{self.ds[device]['mem']['mem_used'] / 1024 ** 2}MB / " \
                          f"{self.ds[device]['mem']['mem_total'] / 1024 ** 2}MB"
            device_texts.append(device_text)
        self.devices_text.set_text('\n'.join(device_texts))
        text_size = self.devices_text.pack()


class LogsBox(urwid.WidgetWrap):
    def __init__(self):
        self.logs_text = urwid.Text('Loading...')
        self.logs_body = urwid.AttrMap(
            urwid.LineBox(urwid.Filler(urwid.AttrMap(self.logs_text, 'logs')),
                          title='Logs',
                          title_align='left'),
            'logs_title'
        )
        super().__init__(self.logs_body)


class TorchStressTUI(urwid.WidgetWrap):
    def __init__(self):
        self.devices_box = DeviceInfoBox()
        self.logs_box = LogsBox()
        self.layout = urwid.Pile([self.devices_box, self.logs_box])
        super().__init__(self.layout)

    def refresh(self, loop, _user_data):
        self.devices_box.refresh()
        # self.logs_box.refresh()
        loop.set_alarm_in(1.0, self.refresh)

    def handle(self, key):
        ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runtime', default=360.0, type=float, help='Stress test runtime length.')
    # TODO add batch size argument (instead of determining a large batch size automatically)
    args = parser.parse_args()

    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.FileHandler('torch-stress.log')],
        force=True,
    )
    logging.debug('Configured logging.')

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    try:
        nvml_init()
        tui = TorchStressTUI()
        main_loop = urwid.MainLoop(tui, palette=palette, unhandled_input=tui.handle)
        main_loop.set_alarm_in(1.0, tui.refresh)
        main_loop.run()
    finally:
        nvml_shutdown()

    #
    # stress_processes_spawn(args)


if __name__ == '__main__':
    main()
