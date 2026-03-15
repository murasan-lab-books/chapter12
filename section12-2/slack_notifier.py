#!/usr/bin/env python3

import os
from typing import Optional
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# 環境変数を.envファイルから読み込む
load_dotenv()


def validate_config() -> dict:
    """
    Slack通知に必要な設定を検証する。

    Returns:
        dict: 検証結果
            {
                "valid": True/False,
                "token_set": True/False,
                "channel_set": True/False,
                "errors": list[str]
            }
    """
    errors = []

    # Bot User OAuth Tokenの確認
    token = os.getenv("SLACK_BOT_TOKEN")
    token_set = bool(token and token.strip())
    if not token_set:
        errors.append("SLACK_BOT_TOKEN が設定されていません")
    elif not token.startswith("xoxb-"):
        errors.append("SLACK_BOT_TOKEN の形式が正しくありません")

    # チャンネルIDの確認
    channel = os.getenv("SLACK_CHANNEL")
    channel_set = bool(channel and channel.strip())
    if not channel_set:
        errors.append("SLACK_CHANNEL が設定されていません")
    elif not channel.startswith("C"):
        errors.append("SLACK_CHANNEL の形式が正しくありません")

    return {
        "valid": len(errors) == 0,
        "token_set": token_set,
        "channel_set": channel_set,
        "errors": errors
    }


def upload_images(
    file_paths: list[str],
    channel: Optional[str] = None,
    message: Optional[str] = None
) -> dict:
    """
    画像ファイルをSlackにアップロードする。

    Args:
        file_paths: アップロードする画像ファイルのパスリスト
        channel: 送信先チャンネルID（省略時は環境変数を使用）
        message: 画像と一緒に投稿するメッセージ

    Returns:
        dict: 実行結果
            {
                "success": True/False,
                "uploaded_count": int,
                "error": str | None
            }
    """
    # チャンネルIDの決定
    target_channel = channel or os.getenv("SLACK_CHANNEL")
    if not target_channel:
        return {
            "success": False,
            "uploaded_count": 0,
            "error": "チャンネルIDが指定されていません"
        }

    # ファイルの存在確認
    for file_path in file_paths:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "uploaded_count": 0,
                "error": f"ファイルが見つかりません: {file_path}"
            }

    # Botトークンの取得
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        return {
            "success": False,
            "uploaded_count": 0,
            "error": "SLACK_BOT_TOKEN が設定されていません"
        }

    try:
        # WebClientの初期化とファイルアップロード
        client = WebClient(token=token)

        if len(file_paths) == 1:
            # 単一ファイルの場合
            client.files_upload_v2(
                channel=target_channel,
                file=file_paths[0],
                initial_comment=message
            )
        else:
            # 複数ファイルの場合
            file_uploads = [
                {"file": path, "title": os.path.basename(path)}
                for path in file_paths
            ]
            client.files_upload_v2(
                channel=target_channel,
                file_uploads=file_uploads,
                initial_comment=message
            )

        return {
            "success": True,
            "uploaded_count": len(file_paths),
            "error": None
        }

    except SlackApiError as e:
        error_code = e.response.get("error", "unknown_error")
        return {
            "success": False,
            "uploaded_count": 0,
            "error": f"Slack APIエラー: {error_code}"
        }


def send_message(
    message: str,
    channel: Optional[str] = None
) -> dict:
    """
    テキストメッセージをSlackに送信する。

    Args:
        message: 送信するメッセージ
        channel: 送信先チャンネルID

    Returns:
        dict: 実行結果
    """
    target_channel = channel or os.getenv("SLACK_CHANNEL")
    token = os.getenv("SLACK_BOT_TOKEN")

    if not target_channel or not token:
        return {"success": False, "error": "設定が不足しています"}

    try:
        client = WebClient(token=token)
        client.chat_postMessage(channel=target_channel, text=message)
        return {"success": True, "error": None}
    except SlackApiError as e:
        return {"success": False, "error": str(e)}
