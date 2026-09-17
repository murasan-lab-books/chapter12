# 第12章 AIにチャレンジ応用編②動くものを追いかける！AI追跡カメラの作成

本章では、サーボモーターとカメラ、YOLOによる物体検出を組み合わせ、対象物を追いかけるカメラを作ります。

## 収録内容

- `section12-1/servo_control.py`：パン・チルト用サーボモーターを制御するスクリプト
- `section12-1/servo_initial_position_setup.py`：サーボの初期位置を設定するスクリプト
- `section12-2/camera_tracker.py`：検出した対象物を追跡する処理
- `section12-2/main.py`：追跡カメラ全体を動かすメインスクリプト
- `section12-2/slack_notifier.py`：検出結果をSlackへ通知する処理

## ライセンス

サンプルコードはAGPL-3.0ライセンスで提供しています。詳細はLICENSEファイルを参照してください。
