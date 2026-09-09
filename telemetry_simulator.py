"""
telemetry_simulator.py

Generates a realistic simulated telemetry stream matching the exact
packet format from the flight computer firmware (TeensyMk3Code.ino):

    [0xAA] [sensor_id: 1 byte] [payload: 1 or 4 bytes LE] [checksum: 1 byte]

- sensor_id 0x01 (stage) has a 1-byte payload, sent raw.
- every other sensor_id has a 4-byte little-endian int32 payload,
  representing the real value * 100 (matching the firmware's scaling).
- checksum is sensor_id XOR every payload byte.

It simulates a full flight profile (pad -> boost -> coast -> apogee ->
descent -> landing) for sensors: stage, yaw, pitch, roll, accel_x,
accel_y, accel_z, altitude, velocity, temperature, pressure, servo_angle.
A configurable fraction of packets are deliberately corrupted (checksum
mismatch) to exercise app.py's error handling, the way real RF noise
would.

Two ways to run it:

1. Virtual serial pair (default, Linux/macOS only):
   python3 telemetry_simulator.py
   -> creates a pty pair and prints the slave device path. Point app.py
      at it before starting app.py:
        export TRACKER_SERIAL_PORT=/dev/pts/N
        python3 app.py

2. Existing serial device (e.g. a socat-linked pair, or real hardware
   loopback for bench testing):
   python3 telemetry_simulator.py --port /dev/pts/5
"""

import argparse
import math
import os
import random
import struct
import sys
import time

try:
    import pty
except Exception:
    pty = None

TX_START_BYTE = 0xAA

SENSOR_IDS = {
    'stage': 0x01,
    'yaw': 0x02,
    'pitch': 0x03,
    'roll': 0x04,
    'accel_x': 0x05,
    'accel_y': 0x06,
    'accel_z': 0x07,
    'altitude': 0x08,
    'velocity': 0x09,
    'temperature': 0x0A,
    'pressure': 0x0B,
    'servo_angle': 0x0C,
}

STAGE_PAD = 0
STAGE_BOOST = 1
STAGE_COAST = 2
STAGE_DESCENT = 3
STAGE_LANDED = 4

BURNOUT_T = 4.0
APOGEE_T = 20.0
TARGET_APOGEE_M = 9144.0  # ~30,000 ft
GROUND_TEMP_C = 15.0
GROUND_PRESSURE_HPA = 1013.25


def encode_packet(sensor_name, value):
    """Build one wire packet for a sensor reading, per the firmware format."""
    sensor_id = SENSOR_IDS[sensor_name]

    if sensor_name == 'stage':
        payload = struct.pack('<B', int(value) & 0xFF)
    else:
        scaled = int(round(value * 100))
        # int32 range guard so struct.pack doesn't blow up on extreme values
        scaled = max(-2_147_483_648, min(2_147_483_647, scaled))
        payload = struct.pack('<i', scaled)

    chk = sensor_id
    for b in payload:
        chk ^= b

    return bytes([TX_START_BYTE, sensor_id]) + payload + bytes([chk])


def corrupt_packet(packet):
    """Flip a random bit in the payload so the checksum no longer matches."""
    packet = bytearray(packet)
    # index 2 is the first payload byte for every sensor (index 0 = start,
    # 1 = sensor_id); flipping it corrupts the value without touching the
    # start byte, so app.py's framing still finds the packet.
    flip_index = 2
    packet[flip_index] ^= 0xFF
    return bytes(packet)


def flight_state(t):
    """
    Returns a dict of sensor values for time t (seconds since launch).
    Rough, deliberately simple kinematics -- good enough to exercise the
    ground station and fusion logic end-to-end, not flight-accurate.
    """
    if t < 0:
        stage = STAGE_PAD
        alt = 0.0
        vel = 0.0
        accel_z = 9.81  # sitting on the pad, accelerometer reads +1g
        accel_x = random.uniform(-0.05, 0.05)
        accel_y = random.uniform(-0.05, 0.05)
        yaw, pitch, roll = 0.0, 90.0, 0.0
        servo = 0.0

    elif t <= BURNOUT_T:
        stage = STAGE_BOOST
        # roughly constant high thrust accel during boost
        accel_z = 220.0 + random.uniform(-10, 10)
        vel = accel_z * t * 0.5  # crude integration for a plausible number
        alt = 0.5 * accel_z * t * t * 0.5
        accel_x = random.uniform(-3, 3)
        accel_y = random.uniform(-3, 3)
        yaw = 0.0 + random.uniform(-1, 1)
        pitch = 90.0 - t * 1.5  # slight pitch-over
        roll = random.uniform(-5, 5)
        servo = 0.0

    elif t <= APOGEE_T:
        stage = STAGE_COAST
        frac = t / APOGEE_T
        alt = TARGET_APOGEE_M * (1 - (1 - frac) ** 2)
        vel = max(250.0 * (1 - frac), 0.0)
        accel_z = -9.81 + random.uniform(-0.3, 0.3)  # coasting under gravity/drag
        accel_x = random.uniform(-1, 1)
        accel_y = random.uniform(-1, 1)
        yaw = random.uniform(-2, 2)
        pitch = max(80.0 - t, 45.0)
        roll = random.uniform(-3, 3)
        servo = 0.0

    else:
        stage = STAGE_DESCENT
        descent_rate = 6.0
        alt = max(TARGET_APOGEE_M - descent_rate * (t - APOGEE_T), 0.0)
        vel = -descent_rate + random.uniform(-0.5, 0.5)
        accel_z = -9.81 + random.uniform(-0.2, 0.2)
        accel_x = random.uniform(-0.5, 0.5)
        accel_y = random.uniform(-0.5, 0.5)
        yaw = random.uniform(-10, 10)  # swinging under chute
        pitch = 20.0 + random.uniform(-5, 5)
        roll = random.uniform(-15, 15)
        servo = 0.0
        if alt <= 0.0:
            stage = STAGE_LANDED
            alt = 0.0
            vel = 0.0
            accel_z = 0.0
            accel_x = 0.0
            accel_y = 0.0

    temperature = GROUND_TEMP_C - 0.0065 * alt  # standard lapse rate
    pressure = GROUND_PRESSURE_HPA * (1 - 2.25577e-5 * alt) ** 5.25588

    return {
        'stage': stage,
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'altitude': alt,
        'velocity': vel,
        'temperature': temperature,
        'pressure': pressure,
        'servo_angle': servo,
    }


def run(write_fn, rate_hz, duration_s, corrupt_fraction, seed):
    if seed is not None:
        random.seed(seed)

    dt = 1.0 / rate_hz
    t = -3.0  # a few seconds of pad idle before "launch" at t=0
    end_t = duration_s

    print(f"Simulating flight: rate={rate_hz} Hz, duration={duration_s}s, "
          f"corrupt_fraction={corrupt_fraction:.2%}")

    packets_sent = 0
    packets_corrupted = 0

    while t <= end_t:
        readings = flight_state(t)
        for sensor_name, value in readings.items():
            packet = encode_packet(sensor_name, value)

            if random.random() < corrupt_fraction:
                packet = corrupt_packet(packet)
                packets_corrupted += 1

            write_fn(packet)
            packets_sent += 1

        if int(t * rate_hz) % (rate_hz * 2) == 0:
            print(f"  t={t:5.1f}s  stage={readings['stage']}  "
                  f"alt={readings['altitude']:7.1f}m  "
                  f"vel={readings['velocity']:6.1f}m/s")

        time.sleep(dt)
        t += dt

    print(f"Done. Sent {packets_sent} packets, {packets_corrupted} "
          f"deliberately corrupted ({packets_corrupted/max(packets_sent,1):.2%}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default=None,
                         help='Existing serial device to write to (e.g. /dev/pts/5 or COM12). '
                              'If omitted, a virtual pty pair is created automatically on POSIX.')
    parser.add_argument('--rate', type=float, default=10.0,
                         help='Telemetry rate in Hz per sensor (default 10, matching '
                              'the intended spec -- note the firmware comment says 10Hz '
                              'but TRANSMIT_INTERVAL_MS=1000 actually gives 1Hz; fix that '
                              'in the firmware if 10Hz is really wanted).')
    parser.add_argument('--duration', type=float, default=90.0,
                         help='Simulated flight duration in seconds (default 90).')
    parser.add_argument('--corrupt-fraction', type=float, default=0.03,
                         help='Fraction of packets to deliberately corrupt (default 0.03 = 3%%).')
    parser.add_argument('--seed', type=int, default=None,
                         help='Random seed for reproducible runs.')
    args = parser.parse_args()

    if args.port:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=1)
        write_fn = ser.write
        print(f"Writing simulated telemetry to existing port: {args.port}")
    elif pty is not None:
        master_fd, slave_fd = pty.openpty()
        slave_name = os.ttyname(slave_fd)
        print(f"Created virtual serial port pair.")
        print(f"Slave device (point app.py at this): {slave_name}")
        print(f"  export TRACKER_SERIAL_PORT={slave_name}")
        print(f"  python3 app.py")
        print("Waiting 3 seconds before starting simulated telemetry...")
        time.sleep(3)
        write_fn = lambda data: os.write(master_fd, data)
    else:
        print("Windows does not expose POSIX pty/termios virtual serial pairs.")
        print("Use --port with an existing serial device, for example:")
        print("  python telemetry_simulator.py --port COM12")
        print("or create a loopback serial pair and point app.py at it.")
        return

    try:
        run(write_fn, args.rate, args.duration, args.corrupt_fraction, args.seed)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
