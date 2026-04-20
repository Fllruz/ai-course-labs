# -*- coding: utf-8 -*-
"""
API клиент для интеграции с внешними сервисами видеонаблюдения
Лабораторная работа №4
Автор: Ходжиев Фируз Фарходович
"""
import os
import json
import requests
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Класс обнаруженного объекта"""
    object_id: str
    object_type: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    center: List[float]  # [x, y]
    timestamp: str


@dataclass
class TrackingResult:
    """Результат отслеживания"""
    frame_id: str
    camera_id: str
    objects: List[DetectedObject]
    processing_time_ms: float
    timestamp: str


class CameraControlAPI:
    """API для управления PTZ-камерами"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
    
    def set_position(self, camera_id: str, pan: float, tilt: float, zoom: float) -> Dict:
        """Установка позиции камеры"""
        response = requests.post(
            f"{self.base_url}/api/v1/cameras/{camera_id}/position",
            json={"pan": pan, "tilt": tilt, "zoom": zoom},
            timeout=5
        )
        return response.json()
    
    def start_tracking(self, camera_id: str, object_id: str) -> Dict:
        """Начать отслеживание объекта"""
        response = requests.post(
            f"{self.base_url}/api/v1/cameras/{camera_id}/track",
            json={"object_id": object_id},
            timeout=5
        )
        return response.json()
    
    def stop_tracking(self, camera_id: str) -> Dict:
        """Остановить отслеживание"""
        response = requests.post(
            f"{self.base_url}/api/v1/cameras/{camera_id}/stop",
            timeout=5
        )
        return response.json()


class AlertSystemAPI:
    """API для системы оповещения"""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url.rstrip('/')
    
    def send_alert(self, camera_id: str, threat_level: str, object_type: str, description: str) -> Dict:
        """Отправить оповещение о威胁"""
        payload = {
            "camera_id": camera_id,
            "threat_level": threat_level,
            "object_type": object_type,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        response = requests.post(
            f"{self.base_url}/api/v1/alerts",
            json=payload,
            timeout=5
        )
        return response.json()
    
    def get_alerts(self, camera_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Получить историю оповещений"""
        params = {"limit": limit}
        if camera_id:
            params["camera_id"] = camera_id
        
        response = requests.get(
            f"{self.base_url}/api/v1/alerts",
            params=params,
            timeout=5
        )
        return response.json().get("alerts", [])


class VideoStorageAPI:
    """API для хранения видеоданных"""
    
    def __init__(self, base_url: str = "http://localhost:8082"):
        self.base_url = base_url.rstrip('/')
    
    def save_frame(self, camera_id: str, frame_data: str, metadata: Dict) -> str:
        """Сохранить кадр с метаданными"""
        payload = {
            "camera_id": camera_id,
            "frame_data": frame_data,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        response = requests.post(
            f"{self.base_url}/api/v1/frames",
            json=payload,
            timeout=10
        )
        return response.json().get("frame_id")
    
    def get_video_stream_url(self, camera_id: str, start_time: str, end_time: str) -> str:
        """Получить URL видеопотока за период"""
        params = {
            "camera_id": camera_id,
            "start_time": start_time,
            "end_time": end_time
        }
        response = requests.get(
            f"{self.base_url}/api/v1/stream",
            params=params,
            timeout=5
        )
        return response.json().get("stream_url")