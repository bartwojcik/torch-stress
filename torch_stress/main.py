import argparse
import logging
import multiprocessing
from collections import deque
from datetime import datetime

import urwid
from texttable import Texttable

from torch_stress.device_monitor import nvml_shutdown, nvml_init, nvml_device_count, nvml_get_name, \
    nvml_get_temp_thresholds, nvml_get_temp, nvml_get_utilization, nvml_get_mem_usage
from torch_stress.stress import spawn_stress_processes, terminate_stress_processes

# key, font color, background color
palette = [
    ('devices_title', 'dark magenta,bold', ''),
    ('devices', 'light magenta,bold', ''),
    ('logs_title', 'yellow', ''),
    ('logs', 'white', ''),
]

REFRESH_RATE = 0.5


class DeviceInfoBox(urwid.WidgetWrap):
    def __init__(self):
        self.devices_text = urwid.Text('Loading...')
        self.devices_body = urwid.AttrMap(
            urwid.LineBox(urwid.AttrMap(self.devices_text, 'devices'),
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
            # stress stats
            self.ds[device]['errors'] = 0
            self.ds[device]['last_updated'] = datetime.now()
        self.update_readings()

    def update_readings(self):
        self.now = datetime.now()
        for device in range(self._num_devices):
            self.ds[device]['temp'] = nvml_get_temp(device)
            # self.ds[device]['fans'] = nvml_get_fanspeed(device)
            self.ds[device]['util'] = nvml_get_utilization(device)
            self.ds[device]['mem'] = nvml_get_mem_usage(device)

    def update_stress_stats(self, device: int, errors: int):
        self.ds[device]['errors'] += errors
        self.ds[device]['last_updated'] = datetime.now()

    def refresh(self):
        self.update_readings()
        table = Texttable()
        table.set_max_width(max_width=120)
        table.set_deco(Texttable.HEADER | Texttable.HLINES | Texttable.VLINES)
        header = ['GPU', 'Errors', 'Last update', 'Temp', 'Slowdown Temp', 'GPU util', 'mem util', 'mem used',
                  'mem total']
        table.header(header)
        table.set_cols_align(['c'] * len(header))
        table.set_cols_valign(['m'] * len(header))
        for device in range(self._num_devices):
            last_update_elapsed_seconds = (datetime.now() - self.ds[device]['last_updated']).total_seconds()
            table.add_row([self.ds[device]['name'],
                           f"{self.ds[device]['errors']}",
                           f"{last_update_elapsed_seconds:.2f}s",
                           f"{self.ds[device]['temp']['gpu_temp']}C",
                           f"{self.ds[device]['thresholds']['temp_slowdown']}C",
                           f"{self.ds[device]['util']['gpu_util_rate']}%",
                           f"{self.ds[device]['util']['mem_util_rate']}%",
                           f"{self.ds[device]['mem']['mem_used'] / 1024 ** 2}MB",
                           f"{self.ds[device]['mem']['mem_total'] / 1024 ** 2}MB"])
        self.devices_text.set_text(table.draw())


class LogsBox(urwid.WidgetWrap):
    def __init__(self):
        self.logs_text = urwid.Filler(urwid.Text('Loading...'), valign='top')
        self.logs_body = urwid.AttrMap(
            urwid.LineBox(
                urwid.AttrMap(self.logs_text, 'logs'),
                title='Logs',
                title_align='left'
            ),
            'logs_title'
        )
        super().__init__(self.logs_body)
        self.logs = deque(maxlen=256)

    def log(self, time, msg):
        self.logs.append(f'[{time.strftime("%Y-%m-%d %H:%M:%S,%f")}] {msg}')

    def refresh(self):
        text = '\n'.join(reversed(self.logs))
        self.logs_text.body.set_text(text)


class TorchStressTUI(urwid.WidgetWrap):
    def __init__(self, status_queue: multiprocessing.Queue):
        self.status_queue = status_queue
        self.devices_box = DeviceInfoBox()
        self.logs_box = LogsBox()
        self.layout = urwid.Frame(header=self.devices_box, body=self.logs_box)
        super().__init__(self.layout)

    def refresh(self, loop, _user_data):
        # TODO possibly capture stderr/stdout from all child processes and it display here?
        while not self.status_queue.empty():
            # handle message
            header, content = self.status_queue.get()
            if isinstance(header, datetime):
                logging.info(f'[{header.timestamp()}] {content}')
                # TODO replace with an additional logging handler?
                self.logs_box.log(header, content)
            elif isinstance(header, int):
                self.devices_box.update_stress_stats(header, content)
        self.devices_box.refresh()
        self.logs_box.refresh()
        loop.set_alarm_in(REFRESH_RATE, self.refresh)

    def handle(self, key):
        if key in ('q', 'Q'):
            raise urwid.ExitMainLoop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runtime', default=360.0, type=float, help='Stress test runtime length.')
    # TODO add a parameter for the number of stress processes per device
    # TODO add batch size argument (instead of determining the batch size automatically)
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

    try:
        nvml_init()
        processes, status_queue, stop_queue = spawn_stress_processes(args)
        tui = TorchStressTUI(status_queue)
        main_loop = urwid.MainLoop(tui, palette=palette, unhandled_input=tui.handle)
        main_loop.set_alarm_in(1.0, tui.refresh)
        main_loop.run()
    finally:
        nvml_shutdown()
        terminate_stress_processes(processes, stop_queue)


if __name__ == '__main__':
    main()
