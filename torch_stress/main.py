import argparse
import os

import urwid

from torch_stress.device_monitor import nvml_shutdown, nvml_init

# key, font color, background color
palette = [
    ('devices_title', 'dark magenta', ''),
    ('devices', 'light magenta,bold', ''),
    ('logs_title', 'yellow', ''),
    ('logs', 'white', ''),
]


class DeviceInfoBox(urwid.WidgetWrap):
    def __init__(self):
        self.devices_text = urwid.Text('Loading...')
        self.devices_body = urwid.AttrMap(
            urwid.LineBox(urwid.Filler(urwid.AttrMap(self.devices_text, 'devices')),
                          title='devices',
                          title_align='left'),
            'devices_title'
        )
        super().__init__(self.devices_body)

    def refresh(self):
        ...


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


def handle_input(key):
    ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runtime', default=360.0, type=float, help='Stress test runtime length.')
    # TODO add batch size argument (instead of determining a large batch size automatically)
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    try:
        num_devices = nvml_init()
        main_loop = urwid.MainLoop(TorchStressTUI(), palette=palette, unhandled_input=handle_input)
        main_loop.run()
    finally:
        nvml_shutdown()

    # logging.basicConfig(
    #     format=(
    #         '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
    #     ),
    #     level=logging.INFO,
    #     handlers=[logging.StreamHandler()],
    #     force=True,
    # )
    # logging.debug('Configured logging.')
    #
    # stress_processes_spawn(args)


if __name__ == '__main__':
    main()
