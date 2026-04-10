# -*- coding: utf-8 -*-
"""
Специализированный инструмент для дипломной работы
Лабораторная работа №2
Автор: Ходжиев Фируз Фарходович
Специальность: Математическое обеспечение и администрирование информационных систем
Тема диплома: Интеллектуальная система управления видеокамерой с функцией слежения за объектами
"""
from langchain.tools import BaseTool
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Глобальное состояние для хранения параметров сглаживания
# (обходим ограничения Pydantic)
_TRACKING_STATE = {
    'last_pan': 0.0,
    'last_tilt': 0.0,
    'last_zoom': 1.0
}

# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 1: Определите схему входных параметров (для системы видеослежения)
# ═════════════════════════════════════════════════════════════════════════════
class CustomToolInput(BaseModel):
    """
    Схема входных параметров для интеллектуальной системы управления видеокамерой.
    """
    object_type: str = Field(
        description="Тип объекта для слежения (человек, автомобиль, животное, другой)",
        default="человек"
    )
    bounding_box: str = Field(
        description="Координаты ограничивающего прямоугольника в формате 'x1,y1,x2,y2' (например: '100,150,250,350')"
    )
    frame_width: int = Field(
        description="Ширина видеокадра в пикселях",
        default=1920,
        ge=320,
        le=7680
    )
    frame_height: int = Field(
        description="Высота видеокадра в пикселях",
        default=1080,
        ge=240,
        le=4320
    )
    confidence: float = Field(
        description="Уровень уверенности детекции от 0.0 до 1.0",
        default=0.85,
        ge=0.0,
        le=1.0
    )

# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 2: Реализуйте класс инструмента для системы видеослежения
# ═════════════════════════════════════════════════════════════════════════════
class CustomTool(BaseTool):
    """
    Интеллектуальный инструмент управления PTZ-камерой с функцией слежения.

    Назначение:
    Анализирует данные детекции объекта (тип, координаты bounding box, уверенность)
    и вычисляет параметры управления камерой: панорамирование, наклон, зум.
    Инструмент имитирует работу модуля интеллектуального трекинга.

    Интеграция с дипломом:
    Инструмент является ключевым компонентом интеллектуальной системы управления
    видеокамерой. Может быть расширен для реального управления сервоприводами
    PTZ-камер через протоколы ONVIF или VISCA.
    """

    # Обязательные атрибуты
    name: str = "ptz_camera_tracker"
    description: str = """
    Инструмент интеллектуального управления PTZ-камерой с функцией слежения.
    Используйте для расчёта параметров движения камеры вслед за объектом.

    Параметры:
    - object_type: тип объекта ('человек', 'автомобиль', 'животное', 'другой')
    - bounding_box: координаты цели в формате 'x1,y1,x2,y2'
    - frame_width: ширина кадра в пикселях
    - frame_height: высота кадра в пикселях
    - confidence: уверенность детекции (0-1)

    Возвращает JSON с решением: углы панорамирования, наклона, коэффициент зума.
    """
    args_schema: Type[BaseModel] = CustomToolInput

    # Параметры калибровки камеры (константы)
    CAMERA_PAN_LIMITS: tuple = (-180, 180)
    CAMERA_TILT_LIMITS: tuple = (-90, 90)
    CAMERA_ZOOM_LIMITS: tuple = (1, 30)
    DEAD_ZONE_RATIO: float = 0.05

    def _run(self, tool_input: str = None, **kwargs) -> str:
        """
        Основная логика интеллектуального управления видеокамерой.

        Args:
            tool_input: Входные данные в формате JSON строки или словаря
            **kwargs: Именованные аргументы

        Returns:
            str: Результат управления камерой в читаемом формате
        """
        # Парсинг входных данных
        if tool_input:
            if isinstance(tool_input, str):
                try:
                    params = json.loads(tool_input)
                except json.JSONDecodeError:
                    # Если не JSON, возможно это словарь в виде строки
                    params = eval(tool_input) if '{' in tool_input else {}
            else:
                params = tool_input
        else:
            params = kwargs
        
        # Извлечение параметров
        object_type = params.get('object_type', 'человек')
        bounding_box = params.get('bounding_box', '0,0,100,100')
        frame_width = params.get('frame_width', 1920)
        frame_height = params.get('frame_height', 1080)
        confidence = params.get('confidence', 0.85)
        
        logger.info(f"🎯 Детекция объекта: {object_type} (уверенность: {confidence:.2f})")

        # Шаг 1: Парсинг и валидация входных данных
        bbox = self._parse_bounding_box(bounding_box)
        if bbox is None:
            return "❌ Ошибка: некорректный формат bounding_box. Используйте 'x1,y1,x2,y2'"

        if not self._validate_coordinates(bbox, frame_width, frame_height):
            return "❌ Ошибка: координаты выходят за пределы кадра"

        # Шаг 2: Фильтрация по уверенности
        if confidence < 0.6:
            logger.warning(f"⚠️ Низкая уверенность детекции ({confidence:.2f}), объект может быть ложным")
            return self._create_low_confidence_response(object_type, confidence)

        # Шаг 3: Вычисление центра объекта
        object_center_x = (bbox[0] + bbox[2]) / 2
        object_center_y = (bbox[1] + bbox[3]) / 2
        object_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # Шаг 4: Расчёт отклонения от центра кадра
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        delta_x = object_center_x - frame_center_x
        delta_y = object_center_y - frame_center_y

        # Шаг 5: Проверка мёртвой зоны
        dead_zone_x = frame_width * self.DEAD_ZONE_RATIO
        dead_zone_y = frame_height * self.DEAD_ZONE_RATIO

        # Шаг 6: Расчёт углов панорамирования и наклона
        pan_angle = self._calculate_pan_angle(delta_x, frame_width)
        tilt_angle = self._calculate_tilt_angle(delta_y, frame_height)

        # Шаг 7: Расчёт коэффициента зума на основе размера объекта
        zoom_factor = self._calculate_zoom(object_size, frame_width, frame_height, object_type)

        # Шаг 8: Применение фильтрации для сглаживания траектории
        smoothed_pan, smoothed_tilt = self._apply_smoothing(pan_angle, tilt_angle)

        # Шаг 9: Формирование результата
        result = {
            "status": "success",
            "object_type": object_type,
            "confidence": confidence,
            "target_center": {"x": round(object_center_x, 2), "y": round(object_center_y, 2)},
            "deviation_from_center": {"dx": round(delta_x, 2), "dy": round(delta_y, 2)},
            "camera_commands": {
                "pan_angle": round(smoothed_pan, 2),
                "tilt_angle": round(smoothed_tilt, 2),
                "zoom_factor": round(zoom_factor, 2)
            },
            "is_in_dead_zone": abs(delta_x) < dead_zone_x and abs(delta_y) < dead_zone_y,
            "needs_movement": abs(delta_x) > dead_zone_x or abs(delta_y) > dead_zone_y,
            "timestamp": datetime.now().isoformat(),
            "algorithm": "PID + Kalman filter (simulated)"
        }

        logger.info(f"🎥 Команды камере: PAN={smoothed_pan:.1f}°, TILT={smoothed_tilt:.1f}°, ZOOM={zoom_factor:.1f}x")
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ═════════════════════════════════════════════════════════════════════════
    # Вспомогательные методы (интеллектуальная логика видеослежения)
    # ═════════════════════════════════════════════════════════════════════════

    def _parse_bounding_box(self, bbox_str: str) -> Optional[tuple]:
        """Парсинг строки с координатами ограничивающего прямоугольника."""
        try:
            parts = bbox_str.split(',')
            if len(parts) != 4:
                return None
            return tuple(int(float(x)) for x in parts)
        except (ValueError, TypeError):
            return None

    def _validate_coordinates(self, bbox: tuple, width: int, height: int) -> bool:
        """Проверка, что координаты находятся в пределах кадра."""
        x1, y1, x2, y2 = bbox
        return (0 <= x1 < x2 <= width) and (0 <= y1 < y2 <= height)

    def _calculate_pan_angle(self, delta_x: float, frame_width: int) -> float:
        """
        Расчёт угла панорамирования на основе отклонения по X.
        Используется пропорциональное управление (P-регулятор).
        """
        # Максимальное отклонение = половина ширины кадра
        max_delta = frame_width / 2
        # Нормализованное отклонение (-1 до 1)
        normalized = max(-1, min(1, delta_x / max_delta)) if max_delta > 0 else 0
        # Преобразование в угол (линейное отображение)
        angle = normalized * self.CAMERA_PAN_LIMITS[1] / 1.5
        return max(self.CAMERA_PAN_LIMITS[0], min(self.CAMERA_PAN_LIMITS[1], angle))

    def _calculate_tilt_angle(self, delta_y: float, frame_height: int) -> float:
        """
        Расчёт угла наклона на основе отклонения по Y.
        """
        max_delta = frame_height / 2
        normalized = max(-1, min(1, delta_y / max_delta)) if max_delta > 0 else 0
        angle = normalized * self.CAMERA_TILT_LIMITS[1] / 1.5
        return max(self.CAMERA_TILT_LIMITS[0], min(self.CAMERA_TILT_LIMITS[1], angle))

    def _calculate_zoom(self, object_size: float, frame_width: int, frame_height: int, object_type: str) -> float:
        """
        Расчёт коэффициента зума на основе размера объекта и его типа.
        """
        global _TRACKING_STATE
        
        frame_area = frame_width * frame_height
        relative_size = object_size / frame_area if frame_area > 0 else 0
        
        # Оптимальный относительный размер объекта в кадре
        optimal_sizes = {
            "человек": 0.15,
            "автомобиль": 0.12,
            "животное": 0.13,
            "другой": 0.10
        }
        optimal_size = optimal_sizes.get(object_type, 0.12)
        
        if relative_size == 0:
            return self.CAMERA_ZOOM_LIMITS[0]
        
        # Расчёт необходимого зума
        zoom_needed = optimal_size / relative_size
        zoom_needed = max(self.CAMERA_ZOOM_LIMITS[0], min(self.CAMERA_ZOOM_LIMITS[1], zoom_needed))
        
        # Применение инерции (изменение зума не более чем на 20% за кадр)
        zoom_change = zoom_needed - _TRACKING_STATE['last_zoom']
        max_change = _TRACKING_STATE['last_zoom'] * 0.2
        if abs(zoom_change) > max_change:
            zoom_needed = _TRACKING_STATE['last_zoom'] + max_change * (1 if zoom_change > 0 else -1)
        
        _TRACKING_STATE['last_zoom'] = zoom_needed
        return round(zoom_needed, 2)

    def _apply_smoothing(self, pan_angle: float, tilt_angle: float) -> tuple:
        """
        Симуляция фильтрации траектории (аналог фильтра Калмана).
        Сглаживает резкие движения камеры.
        """
        global _TRACKING_STATE
        
        smoothing_factor = 0.7  # Коэффициент сглаживания (0-1, чем выше, тем плавнее)
        
        smoothed_pan = _TRACKING_STATE['last_pan'] * smoothing_factor + pan_angle * (1 - smoothing_factor)
        smoothed_tilt = _TRACKING_STATE['last_tilt'] * smoothing_factor + tilt_angle * (1 - smoothing_factor)
        
        _TRACKING_STATE['last_pan'] = smoothed_pan
        _TRACKING_STATE['last_tilt'] = smoothed_tilt
        
        return smoothed_pan, smoothed_tilt

    def _create_low_confidence_response(self, object_type: str, confidence: float) -> str:
        """Формирование ответа при низкой уверенности детекции."""
        return json.dumps({
            "status": "warning",
            "object_type": object_type,
            "confidence": confidence,
            "message": "Низкая уверенность детекции. Рекомендуется перепроверка объекта.",
            "camera_commands": {"pan_angle": 0, "tilt_angle": 0, "zoom_factor": 1},
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False, indent=2)

    async def _arun(self, tool_input: str = None, **kwargs) -> str:
        """Асинхронная версия (для высокопроизводительных систем видеонаблюдения)."""
        return self._run(tool_input, **kwargs)

    def to_langchain_tool(self) -> BaseTool:
        """Конвертация в формат LangChain."""
        return self

# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 3: Пример использования (для тестирования)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Настройка логирования для вывода в консоль
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tool = CustomTool()

    print("=" * 60)
    print("Тестирование интеллектуальной системы управления видеокамерой")
    print("Тема диплома: Интеллектуальная система управления видеокамерой с функцией слежения за объектами")
    print("=" * 60)

    # Тест 1: Слежение за человеком (объект в правом верхнем углу)
    result1 = tool.run({
        "object_type": "человек",
        "bounding_box": "1500,200,1750,500",
        "frame_width": 1920,
        "frame_height": 1080,
        "confidence": 0.92
    })
    print("\n🎯 Тест 1 (слежение за человеком в правом верхнем углу):")
    print(result1)

    # Тест 2: Слежение за автомобилем (объект в центре, уверенность высокая)
    result2 = tool.run({
        "object_type": "автомобиль",
        "bounding_box": "850,450,1070,630",
        "frame_width": 1920,
        "frame_height": 1080,
        "confidence": 0.98
    })
    print("\n🎯 Тест 2 (слежение за автомобилем в центре кадра):")
    print(result2)

    # Тест 3: Низкая уверенность детекции
    result3 = tool.run({
        "object_type": "животное",
        "bounding_box": "500,300,600,400",
        "frame_width": 1280,
        "frame_height": 720,
        "confidence": 0.45
    })
    print("\n🎯 Тест 3 (низкая уверенность детекции):")
    print(result3)

    # Тест 4: Некорректные координаты
    result4 = tool.run({
        "object_type": "человек",
        "bounding_box": "abc,def,ghi,jkl",
        "frame_width": 1920,
        "frame_height": 1080,
        "confidence": 0.85
    })
    print("\n🎯 Тест 4 (ошибочный формат координат):")
    print(result4)
    
    # Тест 5: Альтернативный способ вызова (через именованные аргументы)
    result5 = tool.run(
        tool_input={
            "object_type": "автомобиль",
            "bounding_box": "100,200,300,400",
            "frame_width": 1920,
            "frame_height": 1080,
            "confidence": 0.95
        }
    )
    print("\n🎯 Тест 5 (альтернативный способ вызова):")
    print(result5)
    
    # Тест 6: Демонстрация работы сглаживания (последовательные вызовы)
    print("\n" + "=" * 60)
    print("Тест 6: Демонстрация работы фильтра сглаживания")
    print("=" * 60)
    
    # Создаем новый инструмент для демонстрации сглаживания
    tool2 = CustomTool()
    
    # Последовательные позиции объекта
    positions = [
        "100,100,200,200",      # Позиция 1
        "150,150,250,250",      # Позиция 2 (смещение)
        "200,200,300,300",      # Позиция 3 (ещё смещение)
        "250,250,350,350",      # Позиция 4
        "300,300,400,400",      # Позиция 5
    ]
    
    for i, bbox in enumerate(positions, 1):
        result = tool2.run({
            "object_type": "человек",
            "bounding_box": bbox,
            "frame_width": 640,
            "frame_height": 480,
            "confidence": 0.95
        })
        result_data = json.loads(result)
        print(f"\n📹 Кадр {i}: BBox={bbox}")
        print(f"   PAN={result_data['camera_commands']['pan_angle']:.1f}°, TILT={result_data['camera_commands']['tilt_angle']:.1f}°")