from dataclasses import replace

import pytest

from opendbc.can import CANPacker
from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.honda.angle_profiles import ANGLE_PTM_FIRMWARE, angle_profile, safety_param_bits
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import Bus, CAR, DBC, HondaSafetyFlags
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.sunnypilot.car.honda.carstate import ANGLE_OVERRIDE_RELEASE_FRAMES, HondaCarStateFeatures
from opendbc.sunnypilot.car.honda.clarity_eps import angle_ptm_available
from opendbc.sunnypilot.car.honda.tests.test_clarity_eps import e4
from opendbc.sunnypilot.car.honda.values_ext import HondaSafetyFlagsSP
from opendbc.sunnypilot.car.interfaces import setup_interfaces
from opendbc.sunnypilot.car.tests.runtime_config import make_test_car_config, with_honda_config


def angle_config(platform, *versions, alpha_long=False, angle=True):
  firmware = [structs.CarParams.CarFw(ecu="eps", fwVersion=version, address=0x18DA30F1) for version in versions]
  fp = gen_empty_fingerprint()
  config = with_honda_config(make_test_car_config(), clarity_angle_mode=angle)
  cp = CarInterface.get_params(platform, fp, firmware, alpha_long, False, False, interface_config=config)
  cp.carFw = firmware
  sp = CarInterface.get_params_sp(cp, platform, fp, firmware, alpha_long, False, False)
  config.honda.provider.live_tuning = replace(config.honda.provider.live_tuning, override_fade_up_s=0.0)
  return cp, sp, config


@pytest.mark.parametrize("platform,versions", ANGLE_PTM_FIRMWARE.items())
def test_donor_identities_select_angle_and_resolve_dbc(platform, versions):
  for version in versions:
    cp, sp, config = angle_config(platform, version + b"\x00\x00")
    previous_long = cp.openpilotLongitudinalControl
    setup_interfaces(CarInterface, cp, sp, config)
    assert cp.steerControlType == structs.CarParams.SteerControlType.angle
    assert cp.openpilotLongitudinalControl == previous_long
    profile = angle_profile(cp)
    assert sp.safetyParam & HondaSafetyFlagsSP.CLARITY_ANGLE
    assert sp.safetyParam & HondaSafetyFlagsSP.ANGLE_PROFILE_MASK == (safety_param_bits(profile) if profile else 0)
    packer = CANPacker(DBC[platform][Bus.pt])
    assert {"EPS_LINEAR_ANGLE", "EPS_CAL_STATUS", "STEERING_CONTROL"} <= packer.dbc.name_to_msg.keys()


@pytest.mark.parametrize("versions,expected", [
  ((b"39990-TLA-A220",), False),
  ((b"39990-TLA,A220",), False),
  ((b"39990+TLA+A220", b"39990-TLA-A220"), False),
  ((b"39990+TLA+X999",), False),
  ((b"39990+TLA+A220", b"\x0c auxiliary EPS firmware response"), True),
])
def test_crv_angle_requires_exact_unambiguous_software_id(versions, expected):
  cp, sp, _ = angle_config(CAR.HONDA_CRV_5G, *versions)
  assert angle_ptm_available(cp, sp) == expected


@pytest.mark.parametrize("alpha_long", [False, True])
@pytest.mark.parametrize("build", [0x0D, 0x0E])
def test_crv_angle_mode_off_uses_stock_torque_calibration_and_confirmed_mode(alpha_long, build):
  cp, sp, config = angle_config(CAR.HONDA_CRV_5G, b"39990+TLA+A220", alpha_long=alpha_long, angle=False)
  stock, _, _ = angle_config(CAR.HONDA_CRV_5G, b"39990-TLA-A220", alpha_long=alpha_long, angle=False)
  setup_interfaces(CarInterface, cp, sp, config)
  assert angle_ptm_available(cp, sp)
  assert cp.steerControlType == structs.CarParams.SteerControlType.torque
  assert not sp.safetyParam & (HondaSafetyFlagsSP.CLARITY_ANGLE | HondaSafetyFlagsSP.ANGLE_PROFILE_MASK)
  assert list(cp.lateralParams.torqueV) == list(stock.lateralParams.torqueV)
  assert list(cp.lateralTuning.pid.kpV) == list(stock.lateralTuning.pid.kpV)
  assert cp.openpilotLongitudinalControl == stock.openpilotLongitudinalControl

  ci = CarInterface(cp, sp, config)
  packer = CANPacker(DBC[CAR.HONDA_CRV_5G][Bus.pt])
  for i, (reported, mode, ready) in enumerate(((build, 1, False), (0xFF, 0, False), (build, 0, True), (build, 1, False))):
    status = packer.make_can_msg("EPS_CAL_STATUS", 1, {"REQUESTED_MODE": 8, "ACTIVE_MODE": mode, "BUILD_ID": reported})
    ci.update([(1_000_000_000 + i * 20_000_000, [status])])
    assert ci.CS.eps_cal_status.stock_ready is ready
    _, data = e4(ci, True, v=20.0, torque=0.25)
    assert data[3] == 0 and not data[2] & 0x20
    assert bool(data[2] & 0x80) is ready
    assert (data[:2] != b"\x00\x00") is ready


def test_linear_angle_yields_until_the_driver_releases_below_hysteresis():
  cp, _, config = angle_config(CAR.HONDA_CRV_5G, b"39990+TLA+A220")
  features = HondaCarStateFeatures(cp, config.honda)
  threshold = features.driver_threshold
  assert features.steering_pressed(threshold * 1.1, 50.0, rack_linear_angle=True)
  for _ in range(ANGLE_OVERRIDE_RELEASE_FRAMES):
    assert features.steering_pressed(threshold * 0.5, 50.0, rack_linear_angle=True)
  for _ in range(ANGLE_OVERRIDE_RELEASE_FRAMES - 1):
    assert features.steering_pressed(threshold * 0.35, 50.0, rack_linear_angle=True)
  assert not features.steering_pressed(threshold * 0.35, 50.0, rack_linear_angle=True)
  assert not features.steering_pressed(threshold * 0.5, 50.0)


@pytest.mark.parametrize("alpha_long,bus", [(False, 0), (True, 1)])
@pytest.mark.parametrize("build,idle,drive,yielding", [
  (0x0D, b"\x08\xf1", b"\xe8\xf1", b"\xe4\x01"),
  (0x0E, b"\x0d\x00", b"\x4c\x00", b"\x0c\x00"),
])
def test_crv_angle_requests_require_mode_and_fresh_aligned_sensor(alpha_long, bus, build, idle, drive, yielding):
  cp, sp, config = angle_config(CAR.HONDA_CRV_5G, b"39990+TLA+A220\x00\x00", alpha_long=alpha_long)
  setup_interfaces(CarInterface, cp, sp, config)
  ci = CarInterface(cp, sp, config)
  ci.update([])
  assert CanBus(cp).lkas == bus
  assert e4(ci, True, 5.0, v=20.0)[1][:4] == b"\x00\x00\x08\xf1"

  packer = CANPacker(DBC[CAR.HONDA_CRV_5G][Bus.pt])
  t = 1_000_000_000

  def feed(*, aligned, mode=1, linear=True):
    nonlocal t
    t += 10_000_000
    frames = [packer.make_can_msg("EPS_CAL_STATUS", 1, {"REQUESTED_MODE": 10, "ACTIVE_MODE": mode, "BUILD_ID": build})]
    if linear:
      frames.append(packer.make_can_msg("EPS_LINEAR_ANGLE", 1, {"LINEAR_ANGLE": 2.0, "REF_LATCHED": aligned}))
    return ci.update([(t, frames)])[0]

  assert feed(aligned=True, mode=0).steerFaultTemporary
  assert e4(ci, True, 5.0, measured=2.0, v=20.0)[1][:4] == b"\x00\x00" + idle
  assert feed(aligned=False).steerFaultTemporary
  assert e4(ci, True, 5.0, measured=2.0, v=20.0)[1][:4] == b"\x00\x00" + idle
  state = feed(aligned=True)
  assert state.steeringAngleLinear and state.steeringAngleDeg == 2.0
  _, data = e4(ci, True, 5.0, measured=2.0, v=20.0)
  assert data[2:4] == drive
  _, data = e4(ci, True, 5.0, measured=1.9, v=20.0, pressed=True)
  assert data[2:4] == yielding
  if build == 0x0E:
    assert round(ci.CC.angle_sender_v2.out_angle, 1) == 1.9
    assert ci.CC.angle_sender_v2.active_frames == 0
  for _ in range(6):
    state = feed(aligned=True, linear=False)
  assert state.steerFaultTemporary
  assert e4(ci, True, 5.0, measured=2.0, v=20.0)[1][:4] == b"\x00\x00" + idle


@pytest.mark.parametrize("mode,bus", [(0, 0), (HondaSafetyFlags.BOSCH_LONG, 1)])
@pytest.mark.parametrize("protocol", [2, 3])
def test_crv_panda_blocks_angle_without_alignment_permission_or_freshness(mode, bus, protocol):
  packer = CANPackerSafety(DBC[CAR.HONDA_CRV_5G][Bus.pt])
  safety = libsafety_py.libsafety
  safety.set_current_safety_param_sp(HondaSafetyFlagsSP.CLARITY_ANGLE | (1 << 13))
  assert safety.set_safety_hooks(structs.CarParams.SafetyModel.hondaBosch, mode) == 0
  safety.init_tests()
  safety.set_timer(1_000_000)

  def send(raw=0):
    framing = ({"STEER_CONTROL_ARM": 3, "STEER_CONTROL_ARM_B6": 1} if protocol == 3 else
               {"STEER_TORQUE_REQUEST": 1, "ANGLE_PAYLOAD": 1, "SET_ME_X00_2": 0xF1})
    return safety.safety_tx_hook(packer.make_can_msg_safety("STEERING_CONTROL", bus, {"STEER_TORQUE": raw, **framing}))

  def receive(aligned, sample_bus=1):
    safety.safety_rx_hook(packer.make_can_msg_safety("EPS_LINEAR_ANGLE", sample_bus, {"REF_LATCHED": aligned}))

  safety.set_controls_allowed(True)
  assert not send()
  receive(False)
  assert not send()
  receive(True, 0)
  assert not send()
  receive(True)
  safety.set_controls_allowed(False)
  assert not send()
  safety.set_controls_allowed(True)
  assert not send(2000)
  assert send()
  safety.set_timer(1_050_001)
  assert not send()


@pytest.mark.parametrize("angle", [False, True])
def test_crv_panda_v3_blocks_invalid_markers_and_torque_mode(angle):
  packer = CANPackerSafety(DBC[CAR.HONDA_CRV_5G][Bus.pt])
  safety = libsafety_py.libsafety
  safety.set_current_safety_param_sp((HondaSafetyFlagsSP.CLARITY_ANGLE | (1 << 13)) if angle else 0)
  assert safety.set_safety_hooks(structs.CarParams.SafetyModel.hondaBosch, 0) == 0
  safety.init_tests()

  def tx(**signals):
    return safety.safety_tx_hook(packer.make_can_msg_safety("STEERING_CONTROL", 0, {"STEER_CONTROL_ARM": 3, **signals}))

  assert tx(ANGLE_STANDBY=1) is angle
  assert not tx(ANGLE_STANDBY=1, STEER_TORQUE=1)
  assert not tx(ANGLE_STANDBY=1, SET_ME_X00_2=1)
  safety.set_controls_allowed(True)
  safety.safety_rx_hook(packer.make_can_msg_safety("EPS_LINEAR_ANGLE", 1, {"REF_LATCHED": 1}))
  assert not tx(STEER_TORQUE_REQUEST=1, ANGLE_PAYLOAD=1)
  assert tx(STEER_CONTROL_ARM_B6=1) is angle
