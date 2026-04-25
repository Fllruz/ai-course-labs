# -*- coding: utf-8 -*-
"""
Основная команда агентов (Crew) для исследовательских задач
"""
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import time
import logging
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent
from agents.writer_agent import WriterAgent

logger = logging.getLogger(__name__)


@dataclass
class CrewConfig:
    """Конфигурация команды агентов."""
    name: str = "ResearchCrew"
    process_type: str = "sequential"  # sequential, parallel, hierarchical
    verbose: bool = True
    memory_enabled: bool = True


@dataclass
class CrewResult:
    """Результат работы команды."""
    success: bool
    final_output: str
    agent_results: Dict
    execution_time: float
    timestamp: str


class ResearchCrew:
    """
    Команда агентов для исследовательских задач.

    Архитектура:
    Researcher → Analyst → Writer (последовательный конвейер)

    Процесс выполнения:
    1. Researcher собирает информацию
    2. Analyst обрабатывает и анализирует данные
    3. Writer создаёт финальный отчёт

    Интеграция с дипломом:
    Может быть адаптирована для автоматизации сбора
    и обработки данных для дипломной
    """

    def __init__(self, config: Optional[CrewConfig] = None):
        """
        Инициализация команды агентов.

        Args:
            config: Конфигурация команды
        """
        self.config = config or CrewConfig()

        # Инициализация агентов
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

        # Общий контекст для коммуникации
        self.shared_context = {}

        # Статистика команды
        self.statistics = {
            "crews_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0
        }

        logger.info(f"Команда агентов инициализирована: {self.config.name}")

    def execute(self, task: str, context: Optional[Dict] = None) -> CrewResult:
        """
        Выполнение задачи командой агентов.

        Args:
            task: Описание задачи
            context: Дополнительный контекст

        Returns:
            CrewResult: Результат работы команды
        """
        start_time = time.time()

        logger.info(f"Команда начинает выполнение задачи: {task[:100]}...")

        agent_results = {}
        shared_context = context or {}

        try:
            # ═══════════════════════════════════════════════════════════════
            # ЭТАП 1: ИССЛЕДОВАНИЕ
            # ═══════════════════════════════════════════════════════════════
            logger.info("Этап 1: Исследование")

            researcher_result = self.researcher.execute_task(
                task_description=f"Исследуй тему: {task}",
                context=shared_context
            )
            agent_results["researcher"] = researcher_result

            # Передача результатов аналитику
            shared_context["research_data"] = researcher_result

            # ═══════════════════════════════════════════════════════════════
            # ЭТАП 2: АНАЛИЗ
            # ═══════════════════════════════════════════════════════════════
            logger.info("Этап 2: Анализ")

            analyst_result = self.analyst.execute_task(
                task_description=f"Проанализируй данные по теме: {task}",
                context=shared_context
            )
            agent_results["analyst"] = analyst_result

            # Передача результатов писателю
            shared_context["analysis_data"] = analyst_result

            # ═══════════════════════════════════════════════════════════════
            # ЭТАП 3: НАПИСАНИЕ ОТЧЁТА
            # ═══════════════════════════════════════════════════════════════
            logger.info("Этап 3: Написание отчёта")

            writer_result = self.writer.execute_task(
                task_description=f"Создай отчёт по теме: {task}",
                context=shared_context
            )
            agent_results["writer"] = writer_result

            # ═══════════════════════════════════════════════════════════════
            # ФИНАЛИЗАЦИЯ
            # ═══════════════════════════════════════════════════════════════
            execution_time = time.time() - start_time

            result = CrewResult(
                success=True,
                final_output=writer_result.get("document", ""),
                agent_results=agent_results,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )

            self.statistics["crews_executed"] += 1
            self.statistics["successful_executions"] += 1
            self.statistics["total_execution_time"] += execution_time

            logger.info(f"Команда завершила работу за {execution_time:.2f}с")

            return result

        except Exception as e:
            logger.error(f"Ошибка выполнения команды: {e}", exc_info=True)

            execution_time = time.time() - start_time

            self.statistics["crews_executed"] += 1
            self.statistics["failed_executions"] += 1

            return CrewResult(
                success=False,
                final_output=f"Ошибка: {str(e)}",
                agent_results=agent_results,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )

    def get_statistics(self) -> Dict:
        """Получение статистики команды."""
        return {
            "crew_name": self.config.name,
            "statistics": self.statistics,
            "agent_statistics": {
                "researcher": self.researcher.get_statistics(),
                "analyst": self.analyst.get_statistics(),
                "writer": self.writer.get_statistics()
            }
        }

    def reset_crew(self) -> None:
        """Сброс состояния команды."""
        self.researcher.reset_state()
        self.analyst.reset_state()
        self.writer.reset_state()
        self.shared_context = {}
        logger.info("Команда агентов сброшена")