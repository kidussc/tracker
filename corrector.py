"""Prediction and telemetry fusion for the antenna tracking demonstrator.

Coordinates use local ENU metres: x=east, y=north and z=up.  Azimuth is
clockwise from north; tilt is elevation above the horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path
from typing import Dict, Optional, Sequence


@dataclass
class TrackingConfig:
    antenna_east_m: float = 0.0
    antenna_north_m: float = -500.0
    antenna_height_m: float = 5.0
    telemetry_weight: float = 0.82
    max_prediction_s: float = 10.0


def load_antenna_config(path: str | Path) -> TrackingConfig:
    """Load an antenna location measured from the OpenRocket launch origin.

    The configuration file uses feet so its coordinates match the OpenRocket
    CSV. East and north are positive in their named directions; height is
    positive above the launch point.
    """
    with Path(path).open(encoding='utf-8') as config_file:
        data = json.load(config_file)
    required = {'east_of_launch_ft', 'north_of_launch_ft', 'height_above_launch_ft'}
    missing = required - data.keys()
    if missing:
        raise ValueError(f'Antenna configuration is missing: {sorted(missing)}')
    feet_to_metres = 0.3048
    return TrackingConfig(
        antenna_east_m=float(data['east_of_launch_ft']) * feet_to_metres,
        antenna_north_m=float(data['north_of_launch_ft']) * feet_to_metres,
        antenna_height_m=float(data['height_above_launch_ft']) * feet_to_metres,
    )


def azimuth_tilt(target: Sequence[float], config: TrackingConfig) -> tuple[float, float, float]:
    """Return (azimuth_deg, tilt_deg, range_m) from antenna to a target."""
    east = target[0] - config.antenna_east_m
    north = target[1] - config.antenna_north_m
    up = target[2] - config.antenna_height_m
    horizontal = sqrt(east * east + north * north)
    print(degrees(atan2(east, north)) % 360)
    return (degrees(atan2(east, north)) % 360, degrees(atan2(up, horizontal)), sqrt(horizontal * horizontal + up * up))


def pointing_for_openrocket_row(row: Dict[str, float], config: TrackingConfig) -> Dict[str, float]:
    """Calculate antenna pointing directly from one OpenRocket trajectory row."""
    azimuth, tilt, distance = azimuth_tilt(
        [row['east_m'], row['north_m'], row['altitude_m']], config)
    return {'time_s': row['time_s'], 'azimuth_deg': azimuth,
            'tilt_deg': tilt, 'range_m': distance}


class AntennaCorrector:
    """Fuses an OpenRocket reference point with propagated valid telemetry.

    The propagation intentionally continues for ``max_prediction_s`` after a
    received fix.  After that period it falls back progressively to the flight
    plan rather than pretending an old fix is current.
    """

    def __init__(self, trajectory: Sequence[Dict[str, float]], config: Optional[TrackingConfig] = None):
        self.trajectory = list(trajectory)
        self.config = config or TrackingConfig()
        self.estimate: Optional[list[float]] = None
        self.velocity = [0.0, 0.0, 0.0]
        self.last_time: Optional[float] = None
        self.last_valid_time: Optional[float] = None

    def reference_at(self, time_s: float) -> Dict[str, float]:
        if not self.trajectory:
            raise ValueError("Trajectory is empty")
        if time_s <= self.trajectory[0]["time_s"]:
            return self.trajectory[0]
        for lower, upper in zip(self.trajectory, self.trajectory[1:]):
            if time_s <= upper["time_s"]:
                fraction = (time_s - lower["time_s"]) / (upper["time_s"] - lower["time_s"])
                return {key: lower[key] + (upper[key] - lower[key]) * fraction for key in lower}
        return self.trajectory[-1]

    def update(self, time_s: float, telemetry: Optional[Dict[str, float]] = None) -> Dict[str, float | bool | list[float]]:
        ref = self.reference_at(time_s)
        reference = [ref["east_m"], ref["north_m"], ref["altitude_m"]]
        if self.estimate is None:
            self.estimate = reference[:]
            self.last_time = time_s

        previous_time = self.last_time if self.last_time is not None else time_s
        dt = max(0.0, min(time_s - previous_time, 1.0))
        self.estimate = [self.estimate[i] + self.velocity[i] * dt for i in range(3)]
        valid = telemetry is not None
        if valid:
            if {'position_east_ft', 'position_north_ft', 'altitude_ft'}.issubset(telemetry):
                # The real OpenRocket export provides a GPS-like ENU position,
                # which is more useful than reconstructing it from attitude.
                feet_to_metres = 0.3048
                measured = [telemetry['position_east_ft'] * feet_to_metres,
                            telemetry['position_north_ft'] * feet_to_metres,
                            telemetry['altitude_ft'] * feet_to_metres]
                self.velocity = [0.0, 0.0,
                                 telemetry.get('vertical_velocity_ft_s', 0.0) * feet_to_metres]
            else:
                # Legacy Teensy telemetry encodes ground track in yaw and climb angle in pitch.
                heading, climb = radians(telemetry["yaw"]), radians(telemetry["pitch"])
                speed = telemetry["velocity"]
                horizontal_speed = speed * cos(climb)
                self.velocity = [horizontal_speed * sin(heading), horizontal_speed * cos(heading), speed * sin(climb)]
                measured = [self.estimate[0], self.estimate[1], telemetry["altitude"]]
            weight = max(0.0, min(1.0, self.config.telemetry_weight))
            self.estimate = [weight * measured[i] + (1 - weight) * reference[i] for i in range(3)]
            self.last_valid_time = time_s

        age = time_s - self.last_valid_time if self.last_valid_time is not None else self.config.max_prediction_s
        predicting = not valid and age <= self.config.max_prediction_s
        if not valid and age > self.config.max_prediction_s:
            # Reduce confidence in extrapolation smoothly once its safe window expires.
            plan_weight = min(1.0, (age - self.config.max_prediction_s) / self.config.max_prediction_s)
            self.estimate = [(1 - plan_weight) * self.estimate[i] + plan_weight * reference[i] for i in range(3)]
        self.last_time = time_s
        azimuth, tilt, distance = azimuth_tilt(self.estimate, self.config)
        return {
            "azimuth_deg": azimuth, "tilt_deg": tilt, "range_m": distance,
            "estimated_position": self.estimate[:], "prediction_age_s": max(0.0, age),
            "predicting": predicting, "telemetry_valid": valid,
        }
