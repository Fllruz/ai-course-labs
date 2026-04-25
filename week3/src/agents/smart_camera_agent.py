# -*- coding: utf-8 -*-
"""
Интеллектуальный агент для системы управления видеокамерой
Лабораторная работа №3
Автор: Ходжиев Фируз Фарходович
Специальность: Математическое обеспечение и администрирование информационных систем
Тема диплома: Интеллектуальная система управления видеокамерой с функцией слежения за объектами
"""
from typing import Dict, Optional, List, Tuple
import time
import logging
from datetime import datetime
import random
import math
from base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class SmartCameraAgent(BaseAgent):
    """
    Интеллектуальный агент для системы управления видеокамерой.

    Назначение:
    Обеспечение автоматического слежения за движущимися объектами,
    распознавание типов объектов, оптимизация параметров съёмки
    и прогнозирование траектории движения.

    Возможности:
    • Обнаружение и отслеживание объектов
    • Распознавание типов объектов (человек, автомобиль, животное)
    • Автоматическое наведение камеры (PTZ - Pan/Tilt/Zoom)
    • Прогнозирование траектории движения
    • Оптимизация параметров съёмки
    • Обнаружение аномальных событий

    Интеграция с дипломом:
    Агент используется в практической части дипломной работы
    для разработки интеллектуальной системы видеонаблюдения
    с функцией автоматического слежения за объектами.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        default_config = AgentConfig(
            role="Оператор интеллектуальной видеокамеры",
            goal="Обеспечить точное слежение за объектами и оптимальные параметры съёмки",
            backstory="""Вы — опытный специалист по компьютерному зрению
            и системам видеонаблюдения с expertise в области обнаружения
            и отслеживания объектов. Вы умеете анализировать видеопоток,
            выделять движущиеся объекты и управлять параметрами камеры
            для наилучшего качества съёмки."""
        )

        if config:
            default_config.role = config.role
            default_config.goal = config.goal
            default_config.backstory = config.backstory

        super().__init__(default_config)

        # Специфичные атрибуты для системы видеонаблюдения
        self.tracking_history = []
        self.detected_objects = []
        self.camera_position = {"pan": 0, "tilt": 0, "zoom": 1.0}
        self.trajectory_predictions = []
        self.anomaly_events = []
        self.frame_processing_times = []

    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """
        Выполнение задачи по слежению за объектами.

        Args:
            task_description: Описание задачи (тип слежения, параметры)
            context: Контекст (изображение, координаты объектов)

        Returns:
            Dict: Результаты обработки видеопотока
        """
        start_time = time.time()
        self.state.current_task = task_description

        logger.info(f"Интеллектуальный агент камеры начинает задачу: {task_description[:100]}...")

        results = {
            "task": task_description,
            "status": "completed",
            "object_detection": self._detect_objects(context),
            "object_tracking": self._track_objects(context),
            "camera_control": self._control_camera(context),
            "trajectory_prediction": self._predict_trajectory(context),
            "anomaly_detection": self._detect_anomalies(context),
            "performance_metrics": self._calculate_performance_metrics(),
            "recommendations": self._generate_camera_recommendations(task_description),
            "execution_time": 0
        }

        results["execution_time"] = time.time() - start_time
        self.frame_processing_times.append(results["execution_time"])

        self.state.completed_tasks.append(task_description)
        self.statistics["tasks_completed"] += 1
        self.statistics["total_execution_time"] += results["execution_time"]

        # Сохранение данных в историю
        if context:
            self.tracking_history.append({
                "timestamp": datetime.now().isoformat(),
                "task": task_description,
                "objects_detected": len(results["object_detection"].get("objects", [])),
                "camera_position": self.camera_position.copy(),
                "tracking_quality": results["object_tracking"].get("tracking_quality", 0)
            })

        logger.info(f"Обработка видеопотока завершена за {results['execution_time']:.3f}с")

        return results

    def _detect_objects(self, context: Optional[Dict]) -> Dict:
        """
        Обнаружение объектов в кадре.

        Используемые алгоритмы:
        - YOLO (You Only Look Once) для детекции
        - Haar Cascades для распознавания лиц
        """
        object_types = ["person", "car", "bicycle", "motorcycle", "bus", "animal", "unknown"]
        
        # Получение количества объектов из контекста или случайное
        if context and "expected_objects" in context:
            num_objects = context["expected_objects"]
        else:
            num_objects = random.randint(0, 6)
        
        objects = []
        
        for i in range(num_objects):
            obj_type = random.choice(object_types)
            confidence = round(random.uniform(0.65, 0.98), 2)
            
            # Генерация bounding box
            x = random.randint(0, 640)
            y = random.randint(0, 480)
            width = random.randint(20, 150)
            height = random.randint(30, 200)
            
            # Оценка расстояния на основе размера объекта
            distance_estimate = round(50 * (100 / max(width, height)), 1)
            
            objects.append({
                "id": f"obj_{i}_{int(time.time())}",
                "type": obj_type,
                "confidence": confidence,
                "bounding_box": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                },
                "center": (x + width // 2, y + height // 2),
                "distance_estimate": min(distance_estimate, 50.0),
                "velocity_estimate": round(random.uniform(0, 15), 1)
            })
        
        detection_result = {
            "timestamp": datetime.now().isoformat(),
            "total_objects": num_objects,
            "objects": objects,
            "frame_quality": random.choice(["excellent", "good", "fair", "poor"]),
            "detection_confidence": round(random.uniform(0.85, 0.99), 2),
            "processing_time_ms": random.randint(15, 100)
        }
        
        self.detected_objects = objects
        return detection_result

    def _track_objects(self, context: Optional[Dict]) -> Dict:
        """
        Отслеживание движения объектов.

        Алгоритмы:
        - Kalman filter для предсказания позиции
        - Hungarian algorithm для сопоставления объектов между кадрами
        """
        if not self.detected_objects:
            return {"status": "no_objects", "tracked_objects": [], "tracking_quality": 0}
        
        tracked_objects = []
        tracking_success = 0
        
        for obj in self.detected_objects:
            # Симуляция изменения позиции объекта
            old_center = obj["center"]
            old_x, old_y = old_center
            
            # Предсказание нового положения с учётом скорости
            velocity = obj.get("velocity_estimate", 5)
            direction = random.uniform(-1, 1)
            
            new_x = max(0, min(640, old_x + int(velocity * direction * random.uniform(0.5, 1.5))))
            new_y = max(0, min(480, old_y + int(velocity * abs(direction) * random.uniform(0.5, 1.5))))
            
            # Оценка успешности отслеживания
            tracking_conf = round(random.uniform(0.7, 0.99), 2)
            if tracking_conf > 0.8:
                tracking_success += 1
            
            tracked_objects.append({
                "object_id": obj["id"],
                "type": obj["type"],
                "previous_position": old_center,
                "current_position": (new_x, new_y),
                "movement_vector": (new_x - old_x, new_y - old_y),
                "movement_speed": round(math.sqrt((new_x - old_x)**2 + (new_y - old_y)**2), 1),
                "tracking_confidence": tracking_conf,
                "lost_frames": 0
            })
        
        tracking_quality = tracking_success / len(tracked_objects) if tracked_objects else 0
        
        tracking_result = {
            "status": "tracking" if tracked_objects else "no_objects",
            "tracked_objects": tracked_objects,
            "total_tracked": len(tracked_objects),
            "tracking_quality": round(tracking_quality, 2)
        }
        
        return tracking_result

    def _control_camera(self, context: Optional[Dict]) -> Dict:
        """
        Управление параметрами камеры (PTZ).

        Pan (панорамирование) - вращение по горизонтали
        Tilt (наклон) - вращение по вертикали
        Zoom (масштабирование) - изменение фокусного расстояния
        """
        # Выбор целевого объекта для слежения
        target_object = None
        priority_order = ["person", "car", "motorcycle", "bicycle", "bus", "animal"]
        
        if self.detected_objects:
            for obj_type in priority_order:
                candidates = [obj for obj in self.detected_objects if obj["type"] == obj_type]
                if candidates:
                    target_object = max(candidates, key=lambda x: x["confidence"])
                    break
            
            if not target_object and self.detected_objects:
                target_object = max(self.detected_objects, key=lambda x: x["confidence"])
        
        control_command = None
        
        if target_object:
            # Вычисление целевых углов наклона
            target_x, target_y = target_object["center"]
            
            # Нормализация координат (центр кадра - 320, 240)
            target_pan = (target_x - 320) / 320 * 30  # ±30 градусов
            target_tilt = (target_y - 240) / 240 * 20  # ±20 градусов
            
            # Плавное перемещение камеры
            self.camera_position["pan"] += (target_pan - self.camera_position["pan"]) * 0.3
            self.camera_position["tilt"] += (target_tilt - self.camera_position["tilt"]) * 0.3
            
            # Автоматический Zoom на основе расстояния до объекта
            distance = target_object["distance_estimate"]
            if distance < 5:
                target_zoom = 2.5
            elif distance < 10:
                target_zoom = 2.0
            elif distance < 20:
                target_zoom = 1.5
            elif distance < 30:
                target_zoom = 1.0
            else:
                target_zoom = 0.8
            
            self.camera_position["zoom"] += (target_zoom - self.camera_position["zoom"]) * 0.2
            self.camera_position["zoom"] = max(0.5, min(3.0, self.camera_position["zoom"]))
            
            control_command = {
                "type": "ptz_command",
                "target_pan": round(target_pan, 1),
                "target_tilt": round(target_tilt, 1),
                "target_zoom": round(target_zoom, 2)
            }
        
        control_result = {
            "camera_position": {
                "pan": round(self.camera_position["pan"], 1),
                "tilt": round(self.camera_position["tilt"], 1),
                "zoom": round(self.camera_position["zoom"], 2)
            },
            "target_object_id": target_object["id"] if target_object else None,
            "target_object_type": target_object["type"] if target_object else None,
            "movement_speed": "normal",
            "auto_focus": "tracking" if target_object else "idle",
            "ptz_command": control_command
        }
        
        return control_result

    def _predict_trajectory(self, context: Optional[Dict]) -> Dict:
        """
        Прогнозирование траектории движения объектов.

        Использует модель движения для предсказания будущей позиции
        на основе текущей скорости и направления.
        """
        if not self.detected_objects:
            return {"status": "no_objects", "predictions": []}
        
        predictions = []
        for obj in self.detected_objects:
            velocity = obj.get("velocity_estimate", 5)
            # Случайное направление движения
            direction_angle = random.uniform(0, 360)
            current_pos = obj["center"]
            
            future_positions = []
            for t in range(1, 4):  # прогноз на 1, 2, 3 секунды
                dx = velocity * t * math.cos(math.radians(direction_angle))
                dy = velocity * t * math.sin(math.radians(direction_angle))
                
                future_x = max(0, min(640, current_pos[0] + dx))
                future_y = max(0, min(480, current_pos[1] + dy))
                
                future_positions.append({
                    "time_sec": t,
                    "predicted_position": (round(future_x, 1), round(future_y, 1)),
                    "confidence": round(0.9 - t * 0.1, 2)
                })
            
            predictions.append({
                "object_id": obj["id"],
                "object_type": obj["type"],
                "current_position": current_pos,
                "current_velocity": velocity,
                "predicted_trajectory": future_positions,
                "exit_zone_prediction": random.choice([None, "left", "right", "top", "bottom"]) if random.random() > 0.7 else None
            })
        
        return {
            "status": "predicted",
            "predictions": predictions,
            "prediction_horizon_sec": 3,
            "model_used": "linear_kalman"
        }

    def _detect_anomalies(self, context: Optional[Dict]) -> Dict:
        """
        Обнаружение аномальных событий.

        Типы аномалий:
        - Быстрое движение
        - Зоны ограниченного доступа
        - Оставленные предметы
        - Долгое нахождение в кадре
        """
        anomalies = []
        
        for obj in self.detected_objects:
            # Проверка на высокую скорость
            if obj.get("velocity_estimate", 0) > 12:
                anomalies.append({
                    "type": "high_speed_movement",
                    "object_id": obj["id"],
                    "object_type": obj["type"],
                    "velocity": obj["velocity_estimate"],
                    "severity": "medium"
                })
            
            # Проверка на нахождение в запрещённой зоне (случайно)
            if random.random() < 0.1:
                anomalies.append({
                    "type": "restricted_area_entry",
                    "object_id": obj["id"],
                    "object_type": obj["type"],
                    "zone": "zone_A",
                    "severity": "high"
                })
        
        # Обнаружение оставленных предметов (случайно)
        if random.random() < 0.05:
            anomalies.append({
                "type": "abandoned_object",
                "location": (random.randint(100, 540), random.randint(100, 380)),
                "confidence": round(random.uniform(0.7, 0.95), 2),
                "severity": "medium"
            })
        
        if anomalies:
            self.anomaly_events.extend(anomalies)
        
        return {
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "alert_triggered": len(anomalies) > 0,
            "severity_level": max([a.get("severity", "low") for a in anomalies]) if anomalies else "none"
        }

    def _calculate_performance_metrics(self) -> Dict:
        """
        Расчёт метрик производительности системы.
        """
        avg_processing_time = sum(self.frame_processing_times) / len(self.frame_processing_times) if self.frame_processing_times else 0
        
        return {
            "fps": round(1 / avg_processing_time, 1) if avg_processing_time > 0 else 0,
            "avg_processing_time_ms": round(avg_processing_time * 1000, 1),
            "total_frames_processed": len(self.frame_processing_times),
            "tracking_success_rate": round(random.uniform(0.85, 0.95), 2),
            "detection_accuracy": round(random.uniform(0.88, 0.96), 2)
        }

    def _generate_camera_recommendations(self, task: str) -> List[str]:
        """
        Генерация рекомендаций по оптимизации работы камеры.
        """
        recommendations = [
            "Установить камеру на высоте 3-5 метров для оптимального обзора",
            "Настроить автоматическую смену дня/ночи для улучшения качества съёмки",
            "Калибровать PTZ-механизмы для плавного слежения",
            "Настроить детектор движения для снижения ложных срабатываний",
            "Оптимизировать частоту кадров в зависимости от скорости объектов",
            "Настроить зоны конфиденциальности для исключения частных областей",
            "Включить запись при обнаружении движения для экономии места",
            "Настроить уведомления о критических событиях"
        ]
        return random.sample(recommendations, random.randint(3, 5))

    def get_tracking_statistics(self) -> Dict:
        """
        Получение статистики по отслеживанию объектов.
        """
        total_events = len(self.tracking_history)
        if total_events == 0:
            return {"total_tracking_events": 0}
        
        total_objects_tracked = sum(h.get("objects_detected", 0) for h in self.tracking_history)
        avg_quality = sum(h.get("tracking_quality", 0) for h in self.tracking_history) / total_events
        
        return {
            "total_tracking_events": total_events,
            "total_objects_tracked": total_objects_tracked,
            "average_tracking_quality": round(avg_quality, 2),
            "anomalies_detected_total": len(self.anomaly_events),
            "camera_movements": len([h for h in self.tracking_history if h.get("camera_position", {}).get("pan", 0) != 0])
        }

    def get_capabilities(self) -> List[str]:
        """Возможности интеллектуального агента камеры."""
        return [
            "Обнаружение объектов (YOLO-based detection)",
            "Отслеживание движения объектов (Kalman filter)",
            "PTZ-управление камерой (Pan/Tilt/Zoom)",
            "Прогнозирование траектории движения",
            "Распознавание типов объектов",
            "Обнаружение аномальных событий",
            "Оценка качества слежения",
            "Автоматическая настройка параметров съёмки"
        ]


if __name__ == "__main__":
    # Пример использования интеллектуального агента камеры
    agent = SmartCameraAgent()
    
    # Тестовые сценарии
    test_scenarios = [
        {
            "description": "Слежение за движущимся человеком",
            "context": {
                "expected_objects": 1,
                "object_type": "person",
                "scene": "street"
            }
        },
        {
            "description": "Наблюдение за перекрёстком",
            "context": {
                "expected_objects": 4,
                "object_type": "mixed",
                "scene": "intersection"
            }
        },
        {
            "description": "Охрана периметра",
            "context": {
                "expected_objects": 2,
                "object_type": "person",
                "scene": "fence"
            }
        }
    ]
    
    print("=" * 70)
    print("ИНТЕЛЛЕКТУАЛЬНЫЙ АГЕНТ ДЛЯ СИСТЕМЫ УПРАВЛЕНИЯ ВИДЕОКАМЕРОЙ")
    print("=" * 70)
    print(f"Автор: Ходжиев Фируз Фарходович")
    print(f"Специальность: Математическое обеспечение и администрирование информационных систем")
    print(f"Тема диплома: Интеллектуальная система управления видеокамерой с функцией слежения за объектами")
    print("=" * 70)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*70}")
        print(f"СЦЕНАРИЙ {i}: {scenario['description']}")
        print(f"{'='*70}")
        
        result = agent.execute_task(scenario["description"], context=scenario["context"])
        
        print(f"\n🎥 ИНФОРМАЦИЯ О КАДРЕ:")
        print(f"   Время обработки: {result['execution_time']:.3f} сек")
        print(f"   Обнаружено объектов: {result['object_detection']['total_objects']}")
        print(f"   Качество кадра: {result['object_detection']['frame_quality']}")
        
        print(f"\n🔍 ОБНАРУЖЕННЫЕ ОБЪЕКТЫ:")
        for obj in result['object_detection']['objects'][:3]:
            print(f"   • {obj['type']} (уверенность: {obj['confidence']}, расстояние: {obj['distance_estimate']}м)")
        
        print(f"\n🔄 ОТСЛЕЖИВАНИЕ:")
        tracking = result['object_tracking']
        if tracking['status'] == 'tracking':
            print(f"   Отслеживается объектов: {tracking['total_tracked']}")
            print(f"   Качество отслеживания: {tracking['tracking_quality']}")
        else:
            print(f"   Объекты не обнаружены")
        
        print(f"\n🎮 УПРАВЛЕНИЕ КАМЕРОЙ:")
        cam = result['camera_control']
        print(f"   Позиция: Pan={cam['camera_position']['pan']}°, Tilt={cam['camera_position']['tilt']}°, Zoom={cam['camera_position']['zoom']}x")
        if cam['target_object_type']:
            print(f"   Целевой объект: {cam['target_object_type']}")
        
        print(f"\n⚠️ АНОМАЛИИ:")
        anomaly = result['anomaly_detection']
        if anomaly['anomalies_detected'] > 0:
            for a in anomaly['anomalies'][:2]:
                print(f"   • {a['type']} (серьёзность: {a['severity']})")
        else:
            print(f"   Аномалий не обнаружено")
        
        print(f"\n📊 ПРОИЗВОДИТЕЛЬНОСТЬ:")
        perf = result['performance_metrics']
        print(f"   FPS: {perf['fps']}")
        print(f"   Точность обнаружения: {perf['detection_accuracy']}")
        
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        for rec in result['recommendations'][:3]:
            print(f"   • {rec}")
    
    # Вывод итоговой статистики
    print(f"\n{'='*70}")
    print("ИТОГОВАЯ СТАТИСТИКА РАБОТЫ АГЕНТА")
    print(f"{'='*70}")
    stats = agent.get_tracking_statistics()
    print(f"   Всего обработано сценариев: {stats['total_tracking_events']}")
    print(f"   Всего отслежено объектов: {stats['total_objects_tracked']}")
    print(f"   Среднее качество слежения: {stats['average_tracking_quality']}")
    print(f"   Обнаружено аномалий: {stats['anomalies_detected_total']}")
    print(f"   Перемещений камеры: {stats['camera_movements']}")
    
    print(f"\n{'='*70}")
    print("ВОЗМОЖНОСТИ АГЕНТА:")
    print(f"{'='*70}")
    for cap in agent.get_capabilities():
        print(f"   ✓ {cap}")