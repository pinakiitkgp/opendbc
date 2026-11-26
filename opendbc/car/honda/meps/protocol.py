"""Honda modified-EPS wire protocols v2 (0x0B–0x0D) and v3 (0x0E).

Spec: rwd-xray lab/audit_fable_20261121/v2/PROTOCOL_V2.md (owner-approved slim v2); reference codec meps_v2.py there.

0xE4 byte 3 (openpilot -> EPS)
  AUTH (7:4, 0 = none .. 15 = full) | bits 3:2 = 0 | MODE (1:0: 0 stock, 1 angle). One angle profile, no control bit.
  Byte 2: bit 7 request, bit 5 payload (1 exactly when an angle target is sent).
  Protocol v3 uses byte 2 for angle commands. The stock-ACC Bosch radar drops the v2 mode and payload fields.
  Byte 3 and the torque request bit stay zero. The status layout and rack-angle signal are the same as on 0x0D.
0x6A4 EPS_CAL_STATUS (EPS -> openpilot, 50 Hz)
  layout 4: byte 0 layout (7) | FAULT (6) | STEERING (5) | CLAMP (4) | stock LKAS state (3:2) | active MODE (1:0);
                 byte 2 gain (7:4) | REASON (3:0). Byte 1 is the build ID; bytes 3-6 reference and output.
Layout 4 is told apart by byte 0 bit 7 (a v1 layout-3 frame never sets it; eps_cal_status faults those
"unsupported_build"). It is decoded from the layout-3 DBC signals, which cover byte 0 and byte 2 bits 4:0.
"""
from enum import IntEnum

# Build 0x0C adds the integrator and 100 Hz rack angle. Build 0x0D adds the EPS lock taper with the same wire format.
V2_BUILDS = {0x0B, 0x0C, 0x0D}
V3_BUILDS = {0x0E}
V2_MODE_ANGLE = 1
LINEAR_ANGLE_BUILDS = {0x0C, 0x0D} | V3_BUILDS

V3_ANGLE = 0x0C    # byte 2 arm value 3 selects the angle frame
V3_DRIVE = 0x40    # bit 6 requests steering authority; clear during driver override
V3_STANDBY = 0x0D  # the EPS can select its angle calibration while LKAS is idle


def protocol_of(build_id: int) -> int | None:
  return 3 if build_id in V3_BUILDS else 2 if build_id in V2_BUILDS else None


def v3_byte2(drive: bool) -> int:
  return V3_ANGLE | (V3_DRIVE if drive else 0)


class Reason(IntEnum):
  STEERING = 0
  NO_REQUEST = 1
  MODE_SWITCH_PENDING = 2
  PROTOCOL = 3
  JUMP = 4
  STALE = 5
  ANGLE_INVALID = 6
  AUTHORITY_ZERO = 7
  STOCK_LKAS_NOT_ENGAGED = 8


def byte3_v2(cut: int) -> int:
  """v2: MODE 1 (angle), AUTH = 15 - CUT (AUTH 15 = full authority; the polarity is the opposite of v1's CUT)."""
  return ((15 - (cut & 0xF)) << 4) | V2_MODE_ANGLE


def is_layout4(vl) -> bool:
  return (int(vl["REQUESTED_MODE"]) & 0x8) != 0


def decode_layout4(vl) -> dict:
  """Layout-4 fields from the layout-3 DBC signals of the same bytes."""
  hi = int(vl["REQUESTED_MODE"])                                   # byte 0 bits 7:4 = layout, FAULT, STEERING, CLAMP
  b0_low = (int(vl["ANGLE_LOOP_RUNNING"]) << 3) | int(vl["ACTIVE_MODE"])   # byte 0 bits 3:0
  reason = (int(vl["ANGLE_TARGET_HELD"]) << 3) | (int(vl["ANGLE_FAULT_WHY"]) << 1) | int(vl["ANGLE_FAULT"])
  try:
    reason_name = Reason(reason).name
  except ValueError:
    reason_name = f"REASON_{reason}"
  return dict(fault=bool(hi & 0x4), steering=bool(hi & 0x2), clamp=bool(hi & 0x1), lkas=(b0_low >> 2) & 0x3,
              mode=b0_low & 0x3, build=int(vl["BUILD_ID"]), reason=reason, reason_name=reason_name)
