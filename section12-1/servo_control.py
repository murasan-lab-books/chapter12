#!/usr/bin/env python3
import time
from adafruit_servokit import ServoKit

# ===== 定数 =====
SERVO_CHANNELS = 16
PAN_CHANNEL = 0              # パンサーボ（左右）
TILT_CHANNEL = 1             # チルトサーボ（上下）

# パルス幅（筆者環境で動作確認した値）
PULSE_MIN = 750
PULSE_MAX = 2250

# 動作範囲
PAN_CENTER = 80
PAN_LEFT = 35
PAN_RIGHT = 125

TILT_CENTER = 90
TILT_DOWN = 45
TILT_UP = 120

# 減速制御パラメータ
STEP_ANGLE = 1
NORMAL_DELAY = 0.02
SLOW_DISTANCE = 5
SLOW_DELAY = NORMAL_DELAY * 3


def initialize_servo_kit():
    """ServoKitを初期化し、SG90用のパルス幅を設定"""
    kit = ServoKit(channels=SERVO_CHANNELS)
    kit.servo[PAN_CHANNEL].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
    kit.servo[TILT_CHANNEL].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
    return kit


def _move_smooth(servo, start, end):
    """目標付近で減速しながら滑らかに移動（内部関数）"""
    step = 1 if start < end else -1
    for angle in range(int(start), int(end) + step, step):
        # 目標に近づいたら減速
        if abs(end - angle) <= SLOW_DISTANCE:
            time.sleep(SLOW_DELAY)
        else:
            time.sleep(NORMAL_DELAY)
        servo.angle = angle


def set_pan_angle(kit, angle, smooth=True):
    """パンサーボを指定角度に移動"""
    if not (PAN_LEFT <= angle <= PAN_RIGHT):
        raise ValueError(f"パン角度は{PAN_LEFT}～{PAN_RIGHT}度で指定")

    current = kit.servo[PAN_CHANNEL].angle
    if current is None:
        current = PAN_CENTER

    if smooth:
        _move_smooth(kit.servo[PAN_CHANNEL], current, angle)
    else:
        kit.servo[PAN_CHANNEL].angle = angle


def set_tilt_angle(kit, angle, smooth=True):
    """チルトサーボを指定角度に移動"""
    if not (TILT_DOWN <= angle <= TILT_UP):
        raise ValueError(f"チルト角度は{TILT_DOWN}～{TILT_UP}度で指定")

    current = kit.servo[TILT_CHANNEL].angle
    if current is None:
        current = TILT_CENTER

    if smooth:
        _move_smooth(kit.servo[TILT_CHANNEL], current, angle)
    else:
        kit.servo[TILT_CHANNEL].angle = angle


def set_pan_tilt(kit, pan_angle, tilt_angle, smooth=True):
    """パンとチルトを指定角度に移動"""
    set_pan_angle(kit, pan_angle, smooth)
    set_tilt_angle(kit, tilt_angle, smooth)


def set_center_position(kit, smooth=True):
    """両サーボを中央位置に移動"""
    set_pan_angle(kit, PAN_CENTER, smooth)
    set_tilt_angle(kit, TILT_CENTER, smooth)


def release_servos(kit):
    """サーボを解放（位置保持を停止）"""
    kit.servo[PAN_CHANNEL].fraction = None
    kit.servo[TILT_CHANNEL].fraction = None


def test_pan_servo(kit):
    """パンサーボの動作テスト"""
    print("パンサーボテスト: 左→中央→右→中央")
    set_pan_angle(kit, PAN_LEFT)
    time.sleep(0.5)
    set_pan_angle(kit, PAN_CENTER)
    time.sleep(0.5)
    set_pan_angle(kit, PAN_RIGHT)
    time.sleep(0.5)
    set_pan_angle(kit, PAN_CENTER)
    print("パンサーボテスト完了")


def test_tilt_servo(kit):
    """チルトサーボの動作テスト"""
    print("チルトサーボテスト: 下→中央→上→中央")
    set_tilt_angle(kit, TILT_DOWN)
    time.sleep(0.5)
    set_tilt_angle(kit, TILT_CENTER)
    time.sleep(0.5)
    set_tilt_angle(kit, TILT_UP)
    time.sleep(0.5)
    set_tilt_angle(kit, TILT_CENTER)
    print("チルトサーボテスト完了")


def main():
    """動作テストを実行"""
    print("パン・チルトサーボ制御 動作テスト")
    print(f"パン: {PAN_LEFT}-{PAN_RIGHT}度 (中央:{PAN_CENTER}度)")
    print(f"チルト: {TILT_DOWN}-{TILT_UP}度 (中央:{TILT_CENTER}度)")

    try:
        kit = initialize_servo_kit()
        print("初期化完了")

        set_center_position(kit)
        time.sleep(1)

        test_pan_servo(kit)
        time.sleep(1)

        test_tilt_servo(kit)

        print("\nテスト完了。Ctrl+Cで終了")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n終了します")

    finally:
        release_servos(kit)
        print("サーボを解放しました")


if __name__ == "__main__":
    main()
