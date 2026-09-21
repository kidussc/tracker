"""Run this file to test the production app.py telemetry parser and 3D client."""
from __future__ import annotations

import csv
import random
import struct
from math import atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

from corrector import AntennaCorrector, TrackingConfig

DEFAULTS = {
    'apogee_m': 10_000.0, 'flight_duration_s': 140.0, 'sample_interval_s': 0.25,
    'lateral_drift_m': 1_900.0, 'position_noise_m': 6.0, 'velocity_noise_mps': 1.5,
    # Independent corruption probability for every individual Teensy packet.
    'base_packet_loss': 0.5, 'distance_loss_per_km': 0.008,
    # Set this above zero to create complete multi-second radio outages.
    # It is evaluated once per sample, rather than once per sensor packet.
    'burst_loss_probability': 0.008, 'burst_loss_duration_s': 2.5,
    'seed': 19,
}
CONFIG = {**DEFAULTS, 'antenna_distance_m': 1000.0, 'antenna_bearing_deg': 180.0,
          'antenna_height_m': 5.0, 'aperture_deg': 20.0, 'path_uncertainty_m': 180.0,
          'telemetry_weight': 0.82, 'max_prediction_s': 10.0, 'playback_rate': 3.0}
SENSOR_IDS = {'stage':0x01, 'yaw':0x02, 'pitch':0x03, 'roll':0x04, 'accel_x':0x05,
              'accel_y':0x06, 'accel_z':0x07, 'altitude':0x08, 'velocity':0x09}
OPENROCKET_COLUMNS = {
    'time_s': 'Time (s)', 'altitude_ft': 'Altitude (ft)',
    'vertical_velocity_ft_s': 'Vertical velocity (ft/s)',
    'position_east_ft': 'Position East of launch (ft)',
    'position_north_ft': 'Position North of launch (ft)',
    'latitude_deg': 'Latitude (° N)', 'longitude_deg': 'Longitude (° E)',
    'zenith_deg': 'Vertical orientation (zenith) (°)',
    'azimuth_deg': 'Lateral orientation (azimuth) (°)',
}
OPENROCKET_SENSOR_IDS = {'altitude_ft': 0x0D, 'vertical_velocity_ft_s': 0x0E,
                         'position_east_ft': 0x0F, 'position_north_ft': 0x10,
                         'latitude_deg': 0x11, 'longitude_deg': 0x12,
                         'azimuth_deg': 0x13, 'tilt_deg': 0x14}
SENSOR_SCALES = {'latitude_deg': 1_000_000, 'longitude_deg': 1_000_000}
FEET_TO_METRES = 0.3048


def generate_openrocket_data(path: str | Path, **parameters: float) -> Path:
    p, output = {**DEFAULTS, **parameters}, Path(path)
    rows: List[Dict[str, float]] = []
    for index in range(int(p['flight_duration_s'] / p['sample_interval_s']) + 1):
        time_s = index * p['sample_interval_s']; phase = time_s / p['flight_duration_s']
        rows.append({'time_s':time_s, 'east_m':p['lateral_drift_m'] * phase**1.65,
                     'north_m':450.0 * phase**1.25,
                     'altitude_m':max(0.0, p['apogee_m'] * (1 - (2 * phase - 1)**2))})
    for index, row in enumerate(rows):
        before, after = rows[max(0,index-1)], rows[min(len(rows)-1,index+1)]
        dt = max(after['time_s'] - before['time_s'], .001)
        vx, vy, vz = ((after[key]-before[key])/dt for key in ('east_m','north_m','altitude_m'))
        row.update(velocity_mps=sqrt(vx*vx+vy*vy+vz*vz), yaw_deg=degrees(atan2(vx,vy))%360,
                   pitch_deg=degrees(atan2(vz,sqrt(vx*vx+vy*vy))))
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    return output


def load_trajectory(path: str | Path) -> List[Dict[str, float]]:
    with Path(path).open(newline='', encoding='utf-8') as file:
        return [{key:float(value) for key,value in row.items()} for row in csv.DictReader(file)]


def load_openrocket_trajectory(path: str | Path) -> List[Dict[str, float]]:
    """Load the eight selected fields from an OpenRocket CSV with comment headers."""
    csv_path = Path(path)
    lines = csv_path.read_text(encoding='utf-8-sig').splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith('# Time (s),')), None)
    if header_index is None:
        raise ValueError(f'{csv_path} does not contain an OpenRocket data header')
    reader = csv.DictReader(lines[header_index:], skipinitialspace=True)
    # The column header is a comment only because OpenRocket prefixes it with '# '.
    reader.fieldnames = [name.removeprefix('# ') for name in reader.fieldnames or []]
    missing = set(OPENROCKET_COLUMNS.values()) - set(reader.fieldnames)
    if missing:
        raise ValueError(f'OpenRocket CSV is missing columns: {sorted(missing)}')
    trajectory: List[Dict[str, float]] = []
    for raw in reader:
        # OpenRocket places event comments between the column header and samples.
        if not raw['Time (s)'] or raw['Time (s)'].startswith('#'):
            continue
        values = {name: float(raw[column]) for name, column in OPENROCKET_COLUMNS.items()}
        # OpenRocket gives vertical orientation as zenith; tracker tilt is elevation.
        values['tilt_deg'] = 90.0 - values.pop('zenith_deg')
        values.update(east_m=values['position_east_ft'] * FEET_TO_METRES,
                      north_m=values['position_north_ft'] * FEET_TO_METRES,
                      altitude_m=values['altitude_ft'] * FEET_TO_METRES,
                      velocity_mps=abs(values['vertical_velocity_ft_s']) * FEET_TO_METRES)
        trajectory.append(values)
    if not trajectory:
        raise ValueError(f'{csv_path} contains no simulation samples')
    return trajectory


def encode_teensy_packet(sensor: str, value: float) -> bytes:
    """Exact frame layout used by TeensyMk3Code.ino and app.parse_telemetry_packet."""
    sensor_id = {**SENSOR_IDS, **OPENROCKET_SENSOR_IDS}[sensor]
    payload = (struct.pack('<B', int(value)) if sensor == 'stage' else
               struct.pack('<i', round(value * SENSOR_SCALES.get(sensor, 100))))
    checksum = sensor_id
    for byte in payload: checksum ^= byte
    return bytes([0xAA, sensor_id]) + payload + bytes([checksum])


class TelemetrySimulator:
    def __init__(self, trajectory: Sequence[Dict[str, float]], **parameters: float):
        self.trajectory, self.p = list(trajectory), {**DEFAULTS, **parameters}
        self.random = random.Random(self.p['seed'])
        self.blackout_until_s = -1.0

    def telemetry_at(self, row: Dict[str, float]) -> Dict[str, float]:
        if 'altitude_ft' in row:
            return {sensor: row[sensor] for sensor in OPENROCKET_SENSOR_IDS}
        return {'stage':int(row['time_s'] > 2), 'yaw':row['yaw_deg']+self.random.gauss(0,.45),
                'pitch':row['pitch_deg']+self.random.gauss(0,.35), 'roll':self.random.gauss(0,1),
                'accel_x':0, 'accel_y':0, 'accel_z':-9.81,
                'altitude':max(0,row['altitude_m']+self.random.gauss(0,self.p['position_noise_m'])),
                'velocity':max(0,row['velocity_mps']+self.random.gauss(0,self.p['velocity_noise_mps']))}

    def packet_loss_probability(self, row: Dict[str, float]) -> float:
        distance_km = sqrt(row['east_m']**2+row['north_m']**2+row['altitude_m']**2)/1000
        return min(.95, self.p['base_packet_loss'] + distance_km*self.p['distance_loss_per_km'])

    def frames(self) -> Iterator[Dict[str, object]]:
        for row in self.trajectory:
            if row['time_s'] >= self.blackout_until_s and self.random.random() < self.p['burst_loss_probability']:
                self.blackout_until_s = row['time_s'] + self.p['burst_loss_duration_s']
            yield {'time_s':row['time_s'], 'telemetry':self.telemetry_at(row),
                   'packet_loss_probability':self.packet_loss_probability(row),
                   'blackout':row['time_s'] < self.blackout_until_s}


def run_simulator() -> None:
    """Serve the production app and inject simulated raw bytes into its parser."""
    import app as ground_station
    csv_path = Path(__file__).with_name('openroc.csv')
    if csv_path.exists() and csv_path.stat().st_size:
        trajectory = load_openrocket_trajectory(csv_path)
    else:
        csv_path = Path(__file__).with_name('openrocket_data.csv')
        if not csv_path.exists() or csv_path.stat().st_size == 0: generate_openrocket_data(csv_path, **CONFIG)
        trajectory = load_trajectory(csv_path)

    @ground_station.app.route('/trajectory')
    def trajectory_api():
        from flask import jsonify
        return jsonify({'trajectory':trajectory, 'path_uncertainty_m':CONFIG['path_uncertainty_m']})

    @ground_station.socketio.on('connect')
    def simulator_client_connected():
        ground_station.socketio.emit('tracking_config', CONFIG)

    def sim_loop():
        while True:
            simulator = TelemetrySimulator(trajectory, **CONFIG)
            bearing = radians(CONFIG['antenna_bearing_deg'])
            corrector = AntennaCorrector(trajectory, TrackingConfig(
                antenna_east_m=sin(bearing)*CONFIG['antenna_distance_m'], antenna_north_m=cos(bearing)*CONFIG['antenna_distance_m'],
                antenna_height_m=CONFIG['antenna_height_m'], telemetry_weight=CONFIG['telemetry_weight'], max_prediction_s=CONFIG['max_prediction_s']))
            for frame in simulator.frames():
                parsed = {}
                # Every field is encoded, possibly corrupted, then passed to app.py.
                for sensor, value in frame['telemetry'].items():
                    packet = encode_teensy_packet(sensor, value)
                    if frame['blackout'] or simulator.random.random() < frame['packet_loss_probability']:
                        packet = packet[:-1] + bytes([packet[-1] ^ 0xFF])
                    result = ground_station.parse_telemetry_packet(packet)
                    if result: parsed[result[0]] = result[1]
                required = ({'altitude_ft', 'vertical_velocity_ft_s', 'position_east_ft', 'position_north_ft'}
                            if 'altitude_ft' in frame['telemetry'] else {'yaw', 'pitch', 'altitude', 'velocity'})
                packet_ok = required.issubset(parsed)
                command = corrector.update(frame['time_s'], parsed if packet_ok else None)
                # Keep the text export identical to the command rendered by index.html.
                ground_station.export_tracking_command(frame['time_s'], command)
                event = {'time_s':frame['time_s'], 'packet_ok':packet_ok, 'packet_loss_probability':frame['packet_loss_probability'], 'blackout':frame['blackout'],
                         'command':command, 'config':{'aperture_deg':CONFIG['aperture_deg']}}
                if packet_ok: event['rocket_position'] = command['estimated_position']
                ground_station.socketio.emit('tracking_update', event)
                ground_station.socketio.sleep(CONFIG['sample_interval_s']/CONFIG['playback_rate'])
            ground_station.socketio.sleep(1.5)

    ground_station.socketio.start_background_task(sim_loop)
    ground_station.socketio.run(ground_station.app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_simulator()
