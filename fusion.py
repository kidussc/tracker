"""
fusion.py

Tracker fusion logic, kept separate from the packet parser in app.py.

Takes whatever the parser's shared telemetry state currently holds,
double-integrates accel_x/accel_y into a dead-reckoned position, blends
that toward a trajectory model, and turns the result into a yaw/pitch
command for the tracker (relative to the antenna, accounting for the
antenna-to-launch-tower offset).

app.py is responsible for:
  1. Maintaining the shared `latest` state dict (sensor_name -> value).
  2. Starting `run_loop()` in its own thread, passing in a function that
     returns a snapshot of that state, and a function to emit results
     (e.g. socketio.emit).

Nothing in here touches serial I/O or the packet format at all.
"""

import os
import time
import math

# --- TRACKER GEOMETRY ---
# Antenna is the tracker's own position (origin of its pointing frame).
# The launch tower sits LAUNCH_OFFSET_M away at LAUNCH_BEARING_DEG,
# per the 500 m / ~10 degree offset discussed for this flight.
LAUNCH_OFFSET_M = float(os.environ.get('TRACKER_LAUNCH_OFFSET_M', 500.0))
LAUNCH_BEARING_DEG = float(os.environ.get('TRACKER_LAUNCH_BEARING_DEG', 0.0))
LAUNCH_X = LAUNCH_OFFSET_M * math.sin(math.radians(LAUNCH_BEARING_DEG))
LAUNCH_Y = LAUNCH_OFFSET_M * math.cos(math.radians(LAUNCH_BEARING_DEG))

# Tracker aperture (full width). Half-aperture is the pointing margin
# available on either side of boresight before the target leaves the FOV.
APERTURE_DEG = float(os.environ.get('TRACKER_APERTURE_DEG', 20.0))
HALF_APERTURE_DEG = APERTURE_DEG / 2.0

FUSION_HZ = 10.0
FUSION_DT = 1.0 / FUSION_HZ
# Blend weight applied when pulling the dead-reckoned estimate back toward
# the trajectory model each fusion tick. 0 = ignore model, 1 = snap to model.
MODEL_BLEND_ALPHA = float(os.environ.get('TRACKER_MODEL_BLEND_ALPHA', 0.15))


def trajectory_model(t):
    """
    Placeholder trajectory model: predicted (x, y, altitude) at time t
    (seconds since launch). Replace this with the other team's OpenRocket-
    derived model once you have access to it -- this stub exists so the
    fusion loop has something to blend against in the meantime.

    Uses a simple ascent/descent profile roughly targeting ~9100 m
    (30k ft) apogee, matching the numbers discussed for this flight,
    purely so the blend behavior can be tested end-to-end.
    """
    if t is None or t < 0:
        return 0.0, 0.0, 0.0

    burnout_t = 4.0
    apogee_t = 20.0
    target_apogee_m = 9144.0  # ~30,000 ft

    if t <= apogee_t:
        frac = min(t / apogee_t, 1.0)
        alt = target_apogee_m * (1 - (1 - frac) ** 2)  # decelerating rise
    else:
        descent_rate = 6.0  # m/s
        alt = max(target_apogee_m - descent_rate * (t - apogee_t), 0.0)

    # Simple downrange drift model: a mild constant horizontal drift once
    # the motor burns out, capped near the ~5 km downrange figure discussed.
    if t <= burnout_t:
        x, y = 0.0, 0.0
    else:
        drift_rate = 90.0  # m/s equivalent horizontal drift, tapering below
        dt2 = t - burnout_t
        x = min(drift_rate * dt2 * 0.15, 4500.0)
        y = 0.0

    return x, y, alt


class FusionEngine:
    """
    Holds the dead-reckoning state across ticks and produces one fused
    tracker-pointing update per call to step().
    """

    def __init__(self):
        self.dr_vx = 0.0
        self.dr_vy = 0.0
        self.dr_x = 0.0
        self.dr_y = 0.0
        self.last_fusion_time = None
        self.launch_t0 = None

    def step(self, state, now=None):
        """
        state: dict with at least 'accel_x', 'accel_y', 'altitude', 'stage'
               (missing/None values are handled gracefully).
        now: wall-clock time.time() for this tick (defaults to time.time()).
        Returns a payload dict ready to emit, or None if there isn't enough
        data yet (no accel readings received).
        """
        if now is None:
            now = time.time()

        ax = state.get('accel_x')
        ay = state.get('accel_y')
        alt = state.get('altitude') or 0.0
        stage = state.get('stage')

        if ax is None or ay is None:
            return None

        # Track launch start (t=0) the first time we see a non-zero stage,
        # so the trajectory model has a meaningful time base. Adjust this
        # condition once you know what "stage" actually encodes.
        if self.launch_t0 is None and stage is not None and stage >= 1:
            self.launch_t0 = now

        dt = FUSION_DT if self.last_fusion_time is None else (now - self.last_fusion_time)
        self.last_fusion_time = now

        # --- Dead reckoning: double integration ---
        self.dr_vx += ax * dt
        self.dr_vy += ay * dt
        self.dr_x += self.dr_vx * dt
        self.dr_y += self.dr_vy * dt

        # --- Blend toward trajectory model ---
        t_since_launch = (now - self.launch_t0) if self.launch_t0 else None
        model_x, model_y, model_alt = trajectory_model(t_since_launch)

        fused_x = (1 - MODEL_BLEND_ALPHA) * self.dr_x + MODEL_BLEND_ALPHA * model_x
        fused_y = (1 - MODEL_BLEND_ALPHA) * self.dr_y + MODEL_BLEND_ALPHA * model_y

        # Nudge the integrator's internal position toward the blended
        # result, so drift doesn't just re-accumulate on top next tick.
        self.dr_x, self.dr_y = fused_x, fused_y

        fused_alt = alt if alt else model_alt

        # --- Convert fused position into a yaw/pitch command ---
        # Position relative to the ANTENNA (not the launch tower), since
        # that's the tracker's own pointing origin.
        rel_x = LAUNCH_X + fused_x
        rel_y = LAUNCH_Y + fused_y
        ground_range = math.hypot(rel_x, rel_y)

        yaw_cmd = math.degrees(math.atan2(rel_x, rel_y)) if ground_range > 0.01 else LAUNCH_BEARING_DEG
        slant_range = math.hypot(ground_range, fused_alt)
        pitch_cmd = math.degrees(math.atan2(fused_alt, ground_range)) if slant_range > 0.01 else 0.0

        # --- Aperture margin check ---
        angle_off_boresight = abs(yaw_cmd - LAUNCH_BEARING_DEG)
        margin_deg = HALF_APERTURE_DEG - angle_off_boresight
        margin_ok = margin_deg > 0

        return {
            'fused_x': round(fused_x, 1),
            'fused_y': round(fused_y, 1),
            'fused_alt': round(fused_alt, 1),
            'model_x': round(model_x, 1),
            'model_y': round(model_y, 1),
            'model_alt': round(model_alt, 1),
            'dr_x': round(self.dr_x, 1),
            'dr_y': round(self.dr_y, 1),
            'yaw_cmd': round(yaw_cmd, 2),
            'pitch_cmd': round(pitch_cmd, 2),
            'ground_range': round(ground_range, 1),
            'margin_deg': round(margin_deg, 2),
            'margin_ok': margin_ok,
            't_since_launch': round(t_since_launch, 1) if t_since_launch is not None else None,
        }


def run_loop(get_state, emit, log=None, hz=FUSION_HZ):
    """
    Runs forever at `hz`, pulling a state snapshot via get_state(), running
    it through a FusionEngine, and calling emit(payload) with the result.

    get_state: callable, returns a dict snapshot of current sensor values
               (e.g. a copy of app.py's `latest` dict under its lock).
    emit:      callable, called with the fusion payload dict each tick
               (e.g. socketio.emit('tracker_update', payload)).
    log:       optional callable(msg) for warnings (e.g. logging.warning).
    """
    engine = FusionEngine()
    dt = 1.0 / hz

    while True:
        time.sleep(dt)
        state = get_state()
        payload = engine.step(state)

        if payload is None:
            continue

        emit(payload)

        if log and not payload['margin_ok']:
            log(
                f"APERTURE MARGIN EXCEEDED: yaw={payload['yaw_cmd']:.2f} deg, "
                f"margin={payload['margin_deg']:.2f} deg "
                f"(half-aperture={HALF_APERTURE_DEG:.1f} deg)"
            )
