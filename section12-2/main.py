#!/usr/bin/env python3

import os
import sys
import time
import logging
import signal
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

# 既存モジュールをインポート
import camera_tracker
import slack_notifier


# 環境変数を.envファイルから読み込む
load_dotenv()

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== グローバル変数（シグナルハンドラ用） ====================
_system_running = True


def signal_handler(sig, frame):
    """
    シグナルハンドラ（Ctrl+C対応）

    SIGINT（Ctrl+C）やSIGTERMを受け取った際に、
    安全にシステムを終了するためのフラグを設定する。

    Args:
        sig: シグナル番号
        frame: フレーム情報
    """
    global _system_running
    logger.info("終了シグナルを受信しました。システムを安全に停止します...")
    _system_running = False


# シグナルハンドラを登録
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ==================== メインオーケストレータークラス ====================

class PetMonitoringOrchestrator:
    """
    ペット見守りシステムのオーケストレーター

    常時追跡と定期Slack通知を統合したメインクラス。
    camera_trackerで常時ペットを追跡しながら、タイマー制御で
    定期的に画像を保存してSlackに送信する。

    システムアーキテクチャ:
        1. 常時追跡ループ: camera_tracker.scan_and_track() を継続実行
        2. 定期画像保存: 指定間隔で camera_tracker.capture_images() を実行
        3. タイマートリガー: 保存済み画像を slack_notifier で送信

    Attributes:
        slack_notification_interval: Slack通知間隔（秒）
        image_capture_interval: 画像保存間隔（秒）
        enable_slack: Slack通知の有効/無効
        display: 映像表示フラグ
    """

    def __init__(
        self,
        slack_notification_interval: int = 3600,  # 1時間（秒）
        image_capture_interval: int = 3600,       # 1時間（秒）
        enable_slack: bool = True,
        display: bool = False,
        verbose: bool = False
    ):
        """
        オーケストレーターの初期化

        Args:
            slack_notification_interval: Slack通知間隔（秒）
            image_capture_interval: 画像保存間隔（秒）
            enable_slack: Slack通知を有効化するか
            display: カメラ映像を表示するか
            verbose: 詳細ログを出力するか
        """
        self.slack_notification_interval = slack_notification_interval
        self.image_capture_interval = image_capture_interval
        self.enable_slack = enable_slack
        self.display = display

        # ロギングレベル設定
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # 次回実行時刻を初期化
        self.next_capture_time = datetime.now()
        self.next_notification_time = datetime.now()

        # 環境変数から設定を取得
        self.save_dir = os.getenv("IMAGE_SAVE_DIR", "./captured_images")

        logger.info("=== ペット見守りシステム ===")
        logger.info(f"画像保存間隔: {image_capture_interval // 60}分")
        logger.info(f"Slack通知間隔: {slack_notification_interval // 60}分")
        logger.info(f"Slack通知: {'有効' if enable_slack else '無効'}")
        logger.info(f"映像表示: {'有効' if display else '無効'}")
        logger.info(f"画像保存先: {self.save_dir}")

    def validate_configuration(self) -> bool:
        """
        起動前の設定検証

        必要な環境変数とモジュールの設定を検証する。
        Slack通知が有効な場合は、Slack設定も検証する。

        Returns:
            bool: 設定が有効ならTrue
        """
        logger.info("設定を検証中...")

        # Slack設定の検証（Slack通知が有効な場合のみ）
        if self.enable_slack:
            config = slack_notifier.validate_config()
            if not config["valid"]:
                logger.error("Slack設定が無効です:")
                for error in config["errors"]:
                    logger.error(f"  - {error}")
                logger.error("解決方法: .envファイルでSLACK_BOT_TOKENとSLACK_CHANNELを設定してください")
                return False
            logger.info("Slack設定: OK")
        else:
            logger.info("Slack通知: 無効化されています（テストモード）")

        # 画像保存ディレクトリの検証
        save_path = Path(self.save_dir)
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"画像保存ディレクトリ: {self.save_dir}")
        except Exception as e:
            logger.error(f"画像保存ディレクトリの作成に失敗しました: {e}")
            return False

        logger.info("設定検証: OK")
        return True

    def capture_and_save_images(self) -> list[str]:
        """
        画像をキャプチャして保存する

        camera_tracker.capture_images()を呼び出して、
        現在のカメラ映像から画像を保存する。

        Returns:
            list[str]: 保存した画像ファイルパスのリスト
        """
        logger.info("画像キャプチャを開始します...")

        try:
            file_paths = camera_tracker.capture_images(
                count=3,
                long_edge=800,
                jpeg_quality=70
            )

            if file_paths:
                logger.info(f"{len(file_paths)}枚の画像を保存しました")
                for path in file_paths:
                    logger.debug(f"  - {path}")
            else:
                logger.warning("画像が保存されませんでした")

            return file_paths

        except Exception as e:
            logger.error(f"画像キャプチャエラー: {e}", exc_info=True)
            return []

    def send_to_slack(self, file_paths: list[str]) -> bool:
        """
        画像をSlackに送信する

        保存済みの画像ファイルをSlackにアップロードする。

        Args:
            file_paths: 送信する画像ファイルパスのリスト

        Returns:
            bool: 送信成功ならTrue
        """
        if not self.enable_slack:
            logger.info("Slack通知は無効化されています（スキップ）")
            return True

        if not file_paths:
            logger.warning("送信する画像がありません")
            return False

        logger.info(f"Slackに{len(file_paths)}枚の画像を送信します...")

        try:
            # メッセージ生成
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"ペット見守りシステム - {timestamp}"

            # 画像アップロード
            result = slack_notifier.upload_images(
                file_paths=file_paths,
                message=message
            )

            if result["success"]:
                logger.info(f"Slack送信成功（{result['uploaded_count']}枚）")
                return True
            else:
                logger.error(f"Slack送信失敗: {result['error']}")
                return False

        except Exception as e:
            logger.error(f"Slack送信エラー: {e}", exc_info=True)
            return False

    def run_periodic_tasks(self):
        """
        定期タスクの実行判定

        現在時刻をチェックして、以下のタスクを実行する:
        1. 画像キャプチャタスク（image_capture_interval間隔）
        2. Slack通知タスク（slack_notification_interval間隔）

        タイマー駆動で定期的にこの関数を呼び出すことで、
        スケジュール管理を実現する。
        """
        now = datetime.now()

        # 画像キャプチャタスクの実行判定
        if now >= self.next_capture_time:
            logger.info("=== 定期画像キャプチャ ===")
            file_paths = self.capture_and_save_images()

            # 次回実行時刻を更新
            self.next_capture_time = now + timedelta(seconds=self.image_capture_interval)
            logger.info(f"次回画像キャプチャ: {self.next_capture_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Slack通知タスクの実行判定
        if now >= self.next_notification_time:
            logger.info("=== 定期Slack通知 ===")

            # 通知時に必ず新しい画像をキャプチャする
            # ペット検出の有無に関わらず、その時点の部屋の状況を送信
            logger.info("通知用画像をキャプチャ中...")
            file_paths = camera_tracker.capture_images(
                count=1,
                long_edge=800,
                jpeg_quality=70
            )

            if file_paths and os.path.exists(file_paths[0]):
                # キャプチャした画像をSlackに送信
                self.send_to_slack(file_paths)
            else:
                logger.warning("画像のキャプチャに失敗しました")

            # 次回実行時刻を更新
            self.next_notification_time = now + timedelta(seconds=self.slack_notification_interval)
            logger.info(f"次回Slack通知: {self.next_notification_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def run(self):
        """
        メインループの実行

        処理フロー:
        1. 設定検証
        2. 起動メッセージ送信（Slack有効時）
        3. メインループ:
           - camera_tracker.scan_and_track() で常時追跡
           - 定期タスク実行（画像保存・Slack送信）
        4. 終了処理
        """
        global _system_running

        # 設定検証
        if not self.validate_configuration():
            logger.error("設定検証に失敗しました。システムを終了します")
            return 1

        # 起動メッセージをSlackに送信
        if self.enable_slack:
            startup_message = "ペット見守りシステムを起動しました"
            slack_notifier.send_message(startup_message)

        logger.info("=== システム起動 ===")
        logger.info("常時追跡モードを開始します（Ctrl+Cで終了）")

        try:
            # メインループ
            while _system_running:
                # 定期タスクの実行判定
                self.run_periodic_tasks()

                # 常時追跡の実行
                # continuousモードで実行すると永続ループになるため、
                # 単発実行を繰り返して定期タスクの機会を確保する
                tracking_duration = float(os.getenv("TRACKING_DURATION", "8.0"))
                tracking_fps = float(os.getenv("TRACKING_FPS", "5.0"))
                result = camera_tracker.scan_and_track(
                    scan_steps_pan=9,
                    scan_steps_tilt=5,
                    tracking_duration=tracking_duration,
                    tracking_fps=tracking_fps,
                    continuous=False,  # 単発実行
                    display=self.display,
                    tick_callback=self.run_periodic_tasks
                )

                # qキー終了（display時）
                if result.get("error") == "user_quit":
                    logger.info("qキーにより終了します")
                    break

                # 検出失敗時は短時間待機してから再スキャン
                if not result["detected"]:
                    logger.debug("ペット未検出。再スキャンします...")
                    time.sleep(2.0)

        except KeyboardInterrupt:
            logger.info("ユーザーによる中断")

        except Exception as e:
            logger.error(f"システムエラー: {e}", exc_info=True)
            return 1

        finally:
            # クリーンアップ
            logger.info("システムを終了します...")
            camera_tracker.cleanup()

            # 終了メッセージをSlackに送信
            if self.enable_slack:
                shutdown_message = "ペット見守りシステムを停止しました"
                slack_notifier.send_message(shutdown_message)

            logger.info("システムを正常に終了しました")

        return 0


# ==================== CLIモード ====================

def main():
    """
    CLIモードのメイン処理

    コマンドライン引数を解析してオーケストレーターを実行する。
    """
    parser = argparse.ArgumentParser(
        description="ペット見守りシステム - メインオーケストレーター",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本実行（1時間ごとにSlack通知）
  python main.py

  # Slack通知間隔を30分に設定
  python main.py --interval 30

  # Slack通知を無効化（テスト用）
  python main.py --no-slack

  # 映像表示を有効化
  python main.py --display

  # 詳細ログ出力
  python main.py --verbose

  # 人間も検出対象に追加
  python main.py --classes cat dog person

環境変数:
  SLACK_BOT_TOKEN              Slack Bot User OAuth Token (xoxb-で始まる)
  SLACK_CHANNEL                送信先チャンネルID (C/G/Dで始まる)
  SLACK_NOTIFICATION_INTERVAL  Slack通知間隔（分）デフォルト: 60
  IMAGE_CAPTURE_INTERVAL       画像保存間隔（分）デフォルト: 60
  IMAGE_SAVE_DIR               画像保存ディレクトリ デフォルト: ./captured_images
        """
    )

    # タイマー設定
    parser.add_argument(
        '--interval',
        type=int,
        default=None,
        metavar='MINUTES',
        help='Slack通知間隔（分単位）デフォルト: 環境変数または60分'
    )

    # Slack設定
    parser.add_argument(
        '--no-slack',
        action='store_true',
        help='Slack通知を無効化（テスト用）'
    )

    # 表示設定
    parser.add_argument(
        '--display',
        action='store_true',
        help='カメラ映像をウィンドウに表示'
    )

    # ログ設定
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログを出力'
    )

    # 検出対象クラス
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        metavar='CLASS',
        help='検出対象のクラス名（スペース区切り）デフォルト: cat dog'
    )

    args = parser.parse_args()

    # --display は X サーバ（DISPLAY）が必要。SSH等のheadless環境では自動で無効化する
    if args.display and not os.environ.get("DISPLAY"):
        logger.warning("--display が指定されましたが、DISPLAY が未設定のため無効化します（headless環境）")
        args.display = False

    # Slack通知間隔の決定（優先順位: CLI引数 > 環境変数 > デフォルト）
    if args.interval is not None:
        slack_interval_minutes = args.interval
    else:
        slack_interval_minutes = int(os.getenv("SLACK_NOTIFICATION_INTERVAL", "60"))

    # 画像キャプチャ間隔の取得
    capture_interval_minutes = int(os.getenv("IMAGE_CAPTURE_INTERVAL", "60"))

    # 検出対象クラスの設定（--classesが指定された場合）
    if args.classes:
        camera_tracker.set_target_classes(args.classes)

    # オーケストレーター起動
    orchestrator = PetMonitoringOrchestrator(
        slack_notification_interval=slack_interval_minutes * 60,  # 分→秒
        image_capture_interval=capture_interval_minutes * 60,     # 分→秒
        enable_slack=not args.no_slack,
        display=args.display,
        verbose=args.verbose
    )

    # 実行
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
