import serial
import struct
import threading
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from flask import Flask, send_from_directory
from flask_socketio import SocketIO

_log_handler = RotatingFileHandler('telemetry_link.log', maxBytes=10 * 1024 * 1024,
                                   backupCount=5, encoding='utf-8')
logging.basicConfig(level=logging.INFO, handlers=[_log_handler],
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger('werkzeug').setLevel(logging.ERROR)

SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 115200
TX_START_BYTE = 0xAA
RX_START_BYTE = 0x55
CMD_GROUND_TEST_ON, CMD_LED_ON, CMD_GROUND_TEST_OFF = 0x5A, 0xA5, 0x3C

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f'Connected successfully to {SERIAL_PORT} at {BAUD_RATE} baud.')
except Exception as e:
    print(f'Failed to connect to {SERIAL_PORT}: {e}')
    logging.error('Failed to connect to serial port: %s', e)
    ser = None

SENSOR_MAP = {0x01:'stage', 0x02:'yaw', 0x03:'pitch', 0x04:'roll', 0x05:'accel_x', 0x06:'accel_y', 0x07:'accel_z', 0x08:'altitude', 0x09:'velocity', 0x0A:'temperature', 0x0B:'pressure', 0x0C:'servo_angle',
              0x0D:'altitude_ft', 0x0E:'vertical_velocity_ft_s', 0x0F:'position_east_ft', 0x10:'position_north_ft',
              0x11:'latitude_deg', 0x12:'longitude_deg', 0x13:'azimuth_deg', 0x14:'tilt_deg'}
# Latitude/longitude require finer resolution than the centi-unit scale used by
# the original Teensy fields.  Every sender must use the same scale table.
SENSOR_SCALES = {'latitude_deg': 1_000_000, 'longitude_deg': 1_000_000}

# Text export defaults to the calculated antenna command, which is exactly the
# azimuth and tilt shown on the HTML dashboard.  Set this to 'raw_telemetry'
# to instead export any selected raw fields below.
TEXT_EXPORT_PATH = Path('pointing_values.txt')
TEXT_EXPORT_SOURCE = 'tracking_command'
TEXT_EXPORT_FIELDS = ('azimuth_deg', 'tilt_deg')
TEXT_EXPORT_LABELS = {'azimuth_deg': 'azimuth_deg', 'tilt_deg': 'tilt_deg'}
TEXT_EXPORT_TRIGGER_FIELD = 'tilt_deg'
_latest_telemetry = {}
_telemetry_lock = threading.Lock()


def export_selected_values(updated_sensor):
    """Append a selected, complete telemetry snapshot to a simple text file."""
    if TEXT_EXPORT_SOURCE != 'raw_telemetry' or updated_sensor != TEXT_EXPORT_TRIGGER_FIELD:
        return
    with _telemetry_lock:
        if not all(field in _latest_telemetry for field in TEXT_EXPORT_FIELDS):
            return
        values = ' '.join(
            f'{TEXT_EXPORT_LABELS.get(field, field)}={_latest_telemetry[field]}'
            for field in TEXT_EXPORT_FIELDS
        )
        with TEXT_EXPORT_PATH.open('a', encoding='utf-8') as output:
            output.write(f'{values}\n')


def export_tracking_command(time_s, command):
    """Append the exact pointing command sent to the dashboard, using flight seconds."""
    if TEXT_EXPORT_SOURCE != 'tracking_command':
        return
    with _telemetry_lock:
        with TEXT_EXPORT_PATH.open('a', encoding='utf-8') as output:
            output.write(
                f'time_s={time_s:.2f} azimuth_deg={command["azimuth_deg"]:.2f} '
                f'tilt_deg={command["tilt_deg"]:.2f}\n'
            )


def parse_telemetry_packet(packet):
    if len(packet) < 4 or packet[0] != TX_START_BYTE or packet[1] not in SENSOR_MAP:
        return None
    sensor_id, payload_len = packet[1], (1 if packet[1] == 0x01 else 4)
    if len(packet) != payload_len + 3:
        return None
    payload, received_chk = packet[2:-1], packet[-1]
    calculated_chk = sensor_id
    for byte in payload:
        calculated_chk ^= byte
    if calculated_chk != received_chk:
        logging.warning('Corrupted Packet! ID: 0x%02X | Expected CHK: 0x%02X, Got: 0x%02X', sensor_id, calculated_chk, received_chk)
        return None
    sensor_name = SENSOR_MAP[sensor_id]
    value = (struct.unpack('<B', payload)[0] if sensor_id == 0x01 else
             struct.unpack('<i', payload)[0] / SENSOR_SCALES.get(sensor_name, 100.0))
    with _telemetry_lock:
        _latest_telemetry[sensor_name] = value
    export_selected_values(sensor_name)
    socketio.emit('telemetry_update', {'sensor': sensor_name, 'value': value})
    logging.info('RX Telemetry -> %s: %s', sensor_name.upper(), value)
    return sensor_name, value


def read_serial_data():
    logging.info('Started background serial polling thread.')
    while True:
        if ser and ser.in_waiting >= 3 and ser.read(1)[0] == TX_START_BYTE:
            sensor_id = ser.read(1)[0]
            if sensor_id not in SENSOR_MAP:
                continue
            payload_len = 1 if sensor_id == 0x01 else 4
            while ser.in_waiting < payload_len + 1:
                pass
            parse_telemetry_packet(bytes([TX_START_BYTE, sensor_id]) + ser.read(payload_len + 1))


@app.route('/')
def index():
    return send_from_directory(app.root_path, 'index.html')


@socketio.on('send_command')
def handle_command(data):
    if not ser:
        logging.error('Attempted to send command, but serial port is not connected.')
        return
    cmd_byte = {'GROUND_TEST_ON': CMD_GROUND_TEST_ON, 'LED_ON': CMD_LED_ON, 'GROUND_TEST_OFF': CMD_GROUND_TEST_OFF}.get(data.get('cmd'))
    if cmd_byte is not None:
        ser.write(bytes([RX_START_BYTE, cmd_byte, RX_START_BYTE ^ cmd_byte]))
        logging.info('TX Command -> %s (0x%02X)', data.get('cmd'), cmd_byte)


if __name__ == '__main__':
    threading.Thread(target=read_serial_data, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
