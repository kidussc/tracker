import os
import time
import struct
import threading
import logging
from flask import Flask, render_template, url_for
from flask_socketio import SocketIO

import fusion
import telemetry_simulator

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    filename='telemetry_link.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# CRITICAL FIX: Silence the Flask/Werkzeug HTTP connection spam
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

# --- CONFIGURATION ---
# Overridable via environment variable so this can point at a virtual serial
# port (e.g. from telemetry_simulator.py) without touching code.
SERIAL_PORT = os.environ.get('TRACKER_SERIAL_PORT', '/dev/serial0')
BAUD_RATE = 115200

# Telemetry Protocol (RX from Flight Computer)
TX_START_BYTE = 0xAA

# Command Protocol (TX to Flight Computer)
RX_START_BYTE = 0x55
CMD_GROUND_TEST_ON  = 0x5A
CMD_LED_ON          = 0xA5
CMD_GROUND_TEST_OFF = 0x3C

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

try:
    import serial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    msg = f"Connected successfully to {SERIAL_PORT} at {BAUD_RATE} baud."
    print(msg)
    logging.info(msg)
except Exception as e:
    msg = f"Failed to connect to {SERIAL_PORT}: {e}"
    print(msg)
    logging.error(msg)
    ser = None

SENSOR_MAP = {
    0x01: 'stage',
    0x02: 'yaw',
    0x03: 'pitch',
    0x04: 'roll',
    0x05: 'accel_x',
    0x06: 'accel_y',
    0x07: 'accel_z',
    0x08: 'altitude',
    0x09: 'velocity',
    0x0A: 'temperature',
    0x0B: 'pressure',
    0x0C: 'servo_angle'
}

# --- SHARED STATE ---
# Latest raw value seen for each sensor. This is the only thing the
# fusion module (fusion.py) reads from -- it has no knowledge of the
# serial link or packet format at all.
state_lock = threading.Lock()
latest = {name: None for name in SENSOR_MAP.values()}
link_stats = {'packets_ok': 0, 'packets_corrupt': 0}


def get_state_snapshot():
    """Passed into fusion.run_loop() so it never touches `latest` directly."""
    with state_lock:
        return dict(latest)


def emit_tracker_update(payload):
    """Passed into fusion.run_loop() as its emit callback."""
    socketio.emit('tracker_update', payload)


def read_simulated_telemetry():
    """Fallback telemetry source when there is no physical serial port."""
    logging.info("Started simulated telemetry fallback thread.")
    t = -3.0
    rate_hz = 10.0
    dt = 1.0 / rate_hz

    while True:
        readings = telemetry_simulator.flight_state(t)
        for sensor_name, value in readings.items():
            with state_lock:
                latest[sensor_name] = value
                link_stats['packets_ok'] += 1

            socketio.emit('telemetry_update', {'sensor': sensor_name, 'value': value})

        time.sleep(dt)
        t += dt


def read_serial_data():
    logging.info("Started background serial polling thread.")
    while True:
        if ser and ser.in_waiting >= 3:
            if ser.read(1)[0] == TX_START_BYTE:
                sensor_id = ser.read(1)[0]
                
                if sensor_id not in SENSOR_MAP:
                    continue

                is_8bit = (sensor_id == 0x01)
                payload_len = 1 if is_8bit else 4
                
                while ser.in_waiting < payload_len + 1:
                    pass
                    
                payload_bytes = ser.read(payload_len)
                received_chk = ser.read(1)[0]

                calc_chk = sensor_id
                for b in payload_bytes:
                    calc_chk ^= b

                if calc_chk == received_chk:
                    if is_8bit:
                        value = struct.unpack('<B', payload_bytes)[0]
                    else:
                        raw_int = struct.unpack('<i', payload_bytes)[0]
                        value = raw_int / 100.0

                    sensor_name = SENSOR_MAP[sensor_id]

                    # Only addition to the original parser: record the
                    # value so fusion.py has something to read. Everything
                    # else below this line is unchanged from the original.
                    with state_lock:
                        latest[sensor_name] = value
                        link_stats['packets_ok'] += 1

                    # Push data to HTML dashboard
                    socketio.emit('telemetry_update', {'sensor': sensor_name, 'value': value})
                    
                    # Log the valid incoming telemetry
                    logging.info(f"RX Telemetry -> {sensor_name.upper()}: {value}")
                else:
                    with state_lock:
                        link_stats['packets_corrupt'] += 1
                    err_msg = f"Corrupted Packet! ID: 0x{sensor_id:02X} | Expected CHK: 0x{calc_chk:02X}, Got: 0x{received_chk:02X}"
                    print(err_msg)
                    logging.warning(err_msg)
                    socketio.emit('link_stats', dict(link_stats))

@app.route('/')
def index():
    return render_template(
        'index.html',
        aperture_deg=fusion.APERTURE_DEG,
        half_aperture_deg=fusion.HALF_APERTURE_DEG,
        launch_offset_m=fusion.LAUNCH_OFFSET_M,
        launch_bearing_deg=fusion.LAUNCH_BEARING_DEG,
    )

# --- COMMAND LISTENERS FROM HTML UI ---
@socketio.on('send_command')
def handle_command(data):
    if not ser:
        err_msg = "Attempted to send command, but serial port is not connected."
        print(err_msg)
        logging.error(err_msg)
        return

    command_type = data.get('cmd')
    cmd_byte = None

    if command_type == 'GROUND_TEST_ON':
        cmd_byte = CMD_GROUND_TEST_ON
    elif command_type == 'LED_ON':
        cmd_byte = CMD_LED_ON
    elif command_type == 'GROUND_TEST_OFF':
        cmd_byte = CMD_GROUND_TEST_OFF

    if cmd_byte is not None:
        chk = RX_START_BYTE ^ cmd_byte
        packet = bytes([RX_START_BYTE, cmd_byte, chk])
        ser.write(packet)
        
        success_msg = f"TX Command -> {command_type} (0x{cmd_byte:02X})"
        print(success_msg)
        logging.info(success_msg)

if __name__ == '__main__':
    logging.info("Initializing Ground Station Server...")

    if ser is None:
        logging.info("No serial port detected; starting telemetry simulator fallback.")
        sim_thread = threading.Thread(target=read_simulated_telemetry, daemon=True)
        sim_thread.start()
    else:
        serial_thread = threading.Thread(target=read_serial_data, daemon=True)
        serial_thread.start()

    fusion_thread = threading.Thread(
        target=fusion.run_loop,
        args=(get_state_snapshot, emit_tracker_update, logging.warning),
        daemon=True,
    )
    fusion_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
