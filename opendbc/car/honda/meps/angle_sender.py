"""Honda EPS v2 angle and authority sender, ported from Biscuit-Labs openpilot cb28d3977."""
import numpy as np

from opendbc.car import DT_CTRL
from opendbc.car.honda.angle_profiles import AngleProfile, vm_angle_limit
from opendbc.car.honda.values import CAR

ANGLE_PAYLOAD = 0x20          # byte 2 bit 5 (DBC SET_ME_X00 holds byte 2 bits 6:0)
RAW_PER_DEG = -10.0           # 0x14A STEER_ANGLE factor is -0.1

# EPS limits (angle_ref.py v3: JUMP, SLEW, AMAX), speed m/s
_V = [0., 5., 8., 10., 12., 15., 20., 25., 30.]
EPS_JUMP_DEG = [4.0, 4.0, 3.1, 2.5, 1.9, 1.0, 1.0, 1.0, 1.0]                     # per 10 ms
EPS_SLEW_DEG_S = [170.0, 170.0, 170.0, 143.9, 103.1, 69.7, 43.8, 31.8, 25.3]
ANGLE_STEP_DEG = [min(0.6 * j, 0.9 * s * DT_CTRL) for j, s in zip(EPS_JUMP_DEG, EPS_SLEW_DEG_S, strict=True)]
AMAX_V = [5., 8., 10., 12., 16., 20., 25., 30.]
EPS_AMAX_DEG = [379., 160., 109., 80.3, 50.0, 35.3, 25.7, 20.4]
ANGLE_MAX_FRAC = 0.95
# panda's limits (honda.h honda_angle_accel_limit), the standard angle rule of the other angle brands: 3.6 m/s^2 lateral
# accel and 3.6 m/s^3 lateral jerk (opendbc car/lateral.py MAX_LATERAL_ACCEL / MAX_LATERAL_JERK), with the angle for a
# lateral acceleration from the EPS's own clamp table (0.9 x EPS_AMAX_DEG = 3.6 m/s^2), 1/v^2 below 5 and above 30 m/s,
# capped at STEER_ANGLE_MAX. Panda evaluates them at speed - 1 m/s, openpilot at v_ego, so openpilot stays inside.
STD_ACCEL_SCALE = 0.9
STEER_ANGLE_MAX = 500.0       # deg (panda HONDA_ANGLE_MAX); the logged wheel reached 439
# Linear-angle EPS (build 0x0C on): the same 3.6 m/s^2 limit in the EPS's rack-linear units (0x6A2), i.e. the published
# limit above converted through the firmware VGR map, at the EPS clamp table's knots. Generated with panda's copy
# (honda.h honda_angle_accel_limit_linear) by rwd-xray lab/clarity/CLARITY_ANGLE_MODE_0x0C_20261122/gen_linear_limit.py;
# the EPS clamp is never tighter. 1/v^2 above 80 m/s.
# understeer (cautious second car, owner's car), same method (rwd-xray shared notes CLARITY_ARAGON_UNDERSTEER.md)
LINEAR_SPREAD_KUS = (0.00049, 0.00062)
LINEAR_LIMIT_V = [1.0, 4.13, 4.25, 4.5, 4.75, 5., 8., 10., 12., 16., 20., 25., 30., 32.5, 35., 37.5, 40., 45., 50., 60., 80.]
LINEAR_LIMIT_DEG = [578.36, 578.30, 546.10, 486.34, 434.20, 390.11, 156.61, 102.65, 73.63, 44.91, 31.47, 22.85, 18.15,
                    15.45, 13.35, 11.64, 10.22, 8.05, 6.52, 4.53, 2.56]
STEER_ANGLE_MAX_LINEAR = 578.4   # deg (panda HONDA_ANGLE_MAX_LINEAR)
# target trajectory (v2): rate and acceleration limits by speed, engage fade
_VT = [0., 5., 10., 15., 20., 30.]
TARGET_RATE_DEG_S = [150., 150., 100., 60., 40., 25.]
TARGET_ACCEL_DEG_S2 = [450., 600., 450., 300., 225., 180.]    # 1.5 x torque-mode hands-off p99 (11-15 route)
ENGAGE_S, ENGAGE_FLOOR = 1.5, 0.15    # limits ramp 15% -> 100% after the request rises; ENGAGE_S = default engage time
RAW_MAX = 32767
# authority ramp: while the driver overrides, the request stays up and the loop's authority drops
# to AUTH_MIN at once (route 000000ed: HKG's 17-frame fade pulled the wheel toward center under the driver's hands); after
# the release (and on every engage) it climbs back over Override Fade Up: the setting is the time, 0 = full at once.
AUTH_MIN = 0.0
# With the authority hold the authority ramp gives the soft hand-back, so the target's
# rate/accel limits are full at once (owner's drives f4/f7 on 0x09: Fade Up giving a 1.25 s authority climb with the
# target going straight back to the plan felt best; a ~1 s target fade lagged the model after a release, route ed).
HOLD_LIMITS_FADE_S = 0.0
# Stock 0xE4 byte 2 engage arm (E4 field map 11-15, FUN_2940A / FUN_294BE / FUN_29554), sent with the authority hold:
# bits 3:2 = 2 with bit 6 = the Clarity camera's own arm (0xC8): the stock LKAS torque fade ramps in over 0.10 s and out
# over 0.50 s, instead of 1.02 s / 1.02 s with byte 2 = 0. While the driver yields, bits 3:2 = 1: the stock abort, LKAS
# torque fade to 0 within 2 ms, state 3 -> 4 -> 1; it re-arms over 0.10 s once the bits go back to 2 (request up throughout).
# Why: with the request held up the stock inner rate loop stays engaged at R5 = 0 and brakes the driver's wheel motion
# (-Kp*R6, 120-340 counts of assist while the wheel moves); dropping the request instead leaves it for the 1.02 s stock
# fade. Emulator (lab/clarity/OVERRIDE_TUNE_20261121, hoAB2): LKAS term 0 during a press (stock LKAS-off feel),
# press/release steps below the hold alone, no lockout, STEER_STATUS stays NORMAL. Bit 6 stays set (its falling edge while
# engaged re-ramps, FUN_29646). Sent only with the request: a full disengage keeps the usual arm-1 release (1.02 s).
# Panda's angle rule reads only bits 7 and 5.
E4_ARM2 = 0x48         # SET_ME_X00 (byte 2 bits 6:0): bit 6 | bits 3:2 = 2
E4_ARM2_ABORT = 0x44   # bit 6 | bits 3:2 = 1


def stock_arm_bits(hold: bool, yielding: bool) -> int:
  """0xE4 byte 2 arm bits (SET_ME_X00 units) for the authority hold; 0 = openpilot's usual byte 2 (arm 1)."""
  if not hold:
    return 0
  return E4_ARM2_ABORT if yielding else E4_ARM2

STOCK_ARM_IMAGES = (b'39990+TBA+C020\x00\x00', b'39990+TBA+C120\x00\x00', b'39990+TBA+A030\x00\x00', b'39990+TEG+A010\x00\x00',
                    b'39990+TGG+A120\x00\x00', b'39990+TEA+H020\x00\x00', b'39990+TLA+A040\x00\x00', b'39990+TXM+A040\x00\x00',
                    b'39990+TLA+A220\x00\x00',
                    b'39990+TJB+A220\x00\x00', b'39990+THR+A020\x00\x00', b'39990+T5R+C030\x00\x00',
                    b'39990+TG7+A070\x00\x00', b'39990+TG7+A060\x00\x00', b'39990+T6Z+A110\x00\x00')


def stock_arm_bits_supported(CP) -> bool:
  """The stock 0xE4 byte 2 engage-arm field (stock_arm_bits): traced on the Clarity's A020 (E4 field map 11-15) and shown
  identical on every angle image in STOCK_ARM_IMAGES (SH-2A, TJB/THR/T5R, V850 TG7/T6Z). Torque images and unchecked EPS
  images get no arm bits."""
  if CP.carFingerprint == CAR.HONDA_CLARITY:
    return True
  return any(fw.ecu == "eps" and fw.fwVersion in STOCK_ARM_IMAGES for fw in CP.carFw)


def std_angle_limit(v_ego: float, linear: bool = False, profile: AngleProfile | None = None) -> float:
  """The 3.6 m/s^2 steering-wheel angle (deg), panda's absolute limit at this speed (linear: in rack-linear units; a
  profile: panda's vehicle-model row for that car, angle_profiles.vm_angle_limit)."""
  v = max(float(v_ego), 1.0)
  if profile is not None:
    return vm_angle_limit(v, profile)
  if linear:
    lim = float(np.interp(v, LINEAR_LIMIT_V, LINEAR_LIMIT_DEG)) if v <= LINEAR_LIMIT_V[-1] else \
          LINEAR_LIMIT_DEG[-1] * (LINEAR_LIMIT_V[-1] / v) ** 2
    # panda's car-to-car spread scaling (honda.h honda_angle_accel_limit_linear): a second Clarity understeers less
    lim *= (1.0 + LINEAR_SPREAD_KUS[0] * v * v) / (1.0 + LINEAR_SPREAD_KUS[1] * v * v)
    return min(lim, STEER_ANGLE_MAX_LINEAR)
  if v < AMAX_V[0]:
    lim = STD_ACCEL_SCALE * EPS_AMAX_DEG[0] * (AMAX_V[0] / v) ** 2
  elif v > AMAX_V[-1]:
    lim = STD_ACCEL_SCALE * EPS_AMAX_DEG[-1] * (AMAX_V[-1] / v) ** 2
  else:
    lim = STD_ACCEL_SCALE * float(np.interp(v, AMAX_V, EPS_AMAX_DEG))
  return min(lim, STEER_ANGLE_MAX)


def std_angle_step(v_ego: float, linear: bool = False, profile: AngleProfile | None = None) -> float:
  """Per-frame step (deg) from the 3.6 m/s^3 jerk limit: jerk = accel = 3.6, so the 3.6 m/s^2 angle per second."""
  return std_angle_limit(v_ego, linear, profile) * DT_CTRL


def angle_limit(v_ego: float, linear: bool = False, profile: AngleProfile | None = None) -> float:
  # 0.95 x the EPS's 4.0 m/s^2 table (inside the 0x0B 3.6 clamp only below 5 m/s, where 1/v^2 takes over) and never past
  # panda's 3.6 m/s^2 angle
  if profile is not None:   # panda's row; each profile car's EPS clamp is never tighter (angle_profiles.py)
    return std_angle_limit(v_ego, True, profile)
  if linear:   # the 0.95 x 4.0 table only binds below 5 m/s, where the 1/v^2 panda limit is tighter anyway
    return std_angle_limit(v_ego, True)
  return min(ANGLE_MAX_FRAC * float(np.interp(v_ego, AMAX_V, EPS_AMAX_DEG)), std_angle_limit(v_ego))


def angle_to_raw(angle_deg: float) -> int:
  return int(max(-RAW_MAX, min(RAW_MAX, round(angle_deg * RAW_PER_DEG))))


class AngleModeSender:
  """Turns actuators.steeringAngleDeg into the 0xE4 angle payload. update() returns (raw, request, payload_bit).
  profile: a profile car's angle profile (angle_profiles.py): linear from the start, limits from panda's row for it."""
  def __init__(self, profile: AngleProfile | None = None):
    self.profile = profile
    self.apply_angle = 0.0
    self.apply_rate = 0.0
    self.active_frames = 0
    self.sent_active = False
    self.linear = profile is not None   # linear-angle EPS (0x0C on): desired / measured / limits are rack-linear (CarController sets it)
    self.out_angle = 0.0      # the angle actually sent (= apply_angle once the effort fade is done)

  def _step(self, v_ego: float) -> float:
    return min(float(np.interp(v_ego, _V, ANGLE_STEP_DEG)), std_angle_step(v_ego, self.linear, self.profile))

  def follow_wheel(self, measured_deg: float, measured_rate_deg_s: float, v_ego: float) -> int:
    """Track the wheel during v3 YIELD and restart the effort fade on release."""
    self.apply_angle, self.apply_rate, self.active_frames = float(measured_deg), float(measured_rate_deg_s), 0
    step = self._step(v_ego)
    self.out_angle += float(np.clip(measured_deg - self.out_angle, -step, step))
    return angle_to_raw(self.out_angle)

  def update(self, lat_active: bool, desired_deg: float, measured_deg: float, v_ego: float,
             angle_ready: bool, measured_rate_deg_s: float = 0.0, engage_s: float = ENGAGE_S,
             rate_limit_deg_s: float = 0.0, effort_fade: bool = True) -> tuple[int, bool, int]:
    active = bool(lat_active and angle_ready)
    if active and not self.sent_active and abs(float(measured_deg)) > std_angle_limit(v_ego, self.linear, self.profile):
      # no engage while the wheel is beyond panda's 3.6 m/s^2 angle: the first target must be within one step of the
      # wheel and inside the limit, so panda would block every frame (the Hyundai EV9 path gates the request the same
      # way). CarController reports the dropped request to the loop-running interlock, so it does not count.
      active = False
    if not active:
      self.apply_angle, self.apply_rate = float(measured_deg), float(measured_rate_deg_s)
      self.out_angle = float(measured_deg)
      self.active_frames, self.sent_active = 0, False
      return 0, False, 0
    lim = angle_limit(v_ego, self.linear, self.profile)
    # clamp the target, not the output, so the output still moves along the limited trajectory (from a measured angle
    # above the clamp it walks down to it instead of jumping, which the EPS would reject)
    target = float(np.clip(float(desired_deg), -lim, lim))
    fade = 1.0 if engage_s <= 0.0 else min(1.0, ENGAGE_FLOOR + (1.0 - ENGAGE_FLOOR) * self.active_frames * DT_CTRL / engage_s)
    # step: 0.6 x the EPS jump threshold and panda's jerk step (the 3.6 m/s^3 standard)
    step = self._step(v_ego)
    rate_cap = min(float(np.interp(v_ego, _VT, TARGET_RATE_DEG_S)), step / DT_CTRL)
    if rate_limit_deg_s > 0.0:                # Steering Angle Rate Limit (Night Star / Galaxy), 0 = off
      rate_cap = min(rate_cap, rate_limit_deg_s)
    rate_max = fade * rate_cap
    accel_max = fade * float(np.interp(v_ego, _VT, TARGET_ACCEL_DEG_S2))
    err = target - self.apply_angle
    # exact discrete braking curve: the fastest rate from which the trajectory can still stop at the target with
    # accel_max per frame (the continuous sqrt(2*A*e) overshoots by a frame)
    ad = accel_max * DT_CTRL
    brake = ad * (((0.25 + 2.0 * abs(err) / (ad * DT_CTRL)) ** 0.5) - 0.5) if ad > 0 else 0.0
    rate_des = float(np.sign(err)) * min(rate_max, brake, abs(err) / DT_CTRL)
    self.apply_rate += float(np.clip(rate_des - self.apply_rate, -accel_max * DT_CTRL, accel_max * DT_CTRL))
    # rate_max shapes rate_des only, so a wheel already turning faster at engage eases down at accel_max instead of being
    # stopped in one frame; the EPS's own per-frame jump limit stays a hard bound
    step_rate = step / DT_CTRL
    self.apply_rate = float(np.clip(self.apply_rate, -step_rate, step_rate))
    self.apply_angle += self.apply_rate * DT_CTRL
    # effort fade: the EPS loop pushes in proportion to (target - wheel), so sending wheel + w * (trajectory - wheel)
    # scales its effort by w. w eases 0 -> 1 (smoothstep) over engage_s after the request rises, so the loop takes the
    # wheel over gently instead of pulling it onto the trajectory at full stiffness. The sent angle still moves at most
    # the EPS's per-frame jump limit.
    out = self.apply_angle
    if effort_fade and engage_s > 0.0:
      x = min(1.0, (self.active_frames + 1) * DT_CTRL / engage_s)
      w = x * x * (3.0 - 2.0 * x)
      out = float(measured_deg) + w * (self.apply_angle - float(measured_deg))
    if not self.sent_active:
      self.out_angle = float(measured_deg)   # panda: the first target within one step of the measured angle
    std_lim = std_angle_limit(v_ego, self.linear, self.profile)
    self.out_angle = float(np.clip(self.out_angle + np.clip(out - self.out_angle, -step, step), -std_lim, std_lim))
    self.active_frames += 1
    self.sent_active = True
    return angle_to_raw(self.out_angle), True, ANGLE_PAYLOAD


class AuthorityRamp:
  """The angle loop's authority g (0..1) -> CUT = round(15 (1 - g)); byte 3 carries AUTH = 15 - CUT (byte3_v2). The EPS
  slews its own gain toward AUTH (up 3 and down 127 per tick on 0x0B) and zeroes it whenever the loop (re)activates."""
  def __init__(self):
    self.g = 0.0

  def update(self, active: bool, yielding: bool, fade_up_s: float) -> int:
    if not active:
      self.g = 0.0
    elif yielding:
      self.g = AUTH_MIN               # instant (route 000000ed: the 17-frame HKG fade pulled the wheel under the driver's hands)
    else:
      self.g = 1.0 if fade_up_s <= 0.0 else min(1.0, self.g + DT_CTRL / fade_up_s)
    return int(round(15.0 * (1.0 - self.g)))
