#!/usr/bin/env python3
import time
from adafruit_servokit import ServoKit

# 初期化
kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(750, 2250)  # パン
kit.servo[1].set_pulse_width_range(750, 2250)  # チルト

# 中央位置に設定
kit.servo[0].angle = 90
kit.servo[1].angle = 90

print("90度に設定しました。サーボホーンを取り付けてください。")
print("Ctrl+C で終了")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n終了")
