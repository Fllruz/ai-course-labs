# -*- coding: utf-8 -*-
"""
Ядро AI-агента с поддержкой инструментов и памяти
Лабораторная работа №2
Дисциплина: Искусственный интеллект
Автор: Ходжиев Фируз Фарходович
Группа: МОА-221
Специальность: Математическое обеспечение и администрирование информационных систем
Тема диплома: Интеллектуальная система управления видеокамерой с функцией слежения за объектами
Дата: 2026
"""
import os
import sys
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate

# Локальные импорты
from tools.search_tool import SearchTool
from tools.calc_tool import CalculateTool
from tools.custom_tool import CustomTool

# Импорт семантической памяти (если есть)
try:
    from memory.semantic_memory import SemanticMemory
    SEMANTIC_MEMORY_AVAILABLE = True
except ImportError:
    SEMANTIC_MEMORY_AVAILABLE = False
    logging.warning("SemanticMemory не доступен")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Конфигурация AI-агента.

    Атрибуты:
        name: Имя агента
        version: Версия агента
        max_iterations: Максимальное количество итераций ReAct
        temperature: Параметр креативности LLM
        memory_enabled: Флаг включения памяти
        guardrails_enabled: Флаг включения защиты
        verbose: Режим подробного логирования
    """
    name: str = "PTZCameraTrackerAgent"
    version: str = "2.0"
    max_iterations: int = 10
    temperature: float = 0.7
    memory_enabled: bool = True
    guardrails_enabled: bool = False  # Отключаем guardrails, если нет модуля
    verbose: bool = True


@dataclass
class AgentResponse:
    """
    Структурированный ответ агента.

    Атрибуты:
        success: Флаг успешного выполнения
        answer: Текстовый ответ
        steps: Список выполненных шагов
        duration_ms: Время выполнения в миллисекундах
        tokens_used: Оценка использованных токенов
        error: Сообщение об ошибке (если есть)
    """
    success: bool
    answer: str
    steps: List[Dict]
    duration_ms: int
    tokens_used: int
    error: Optional[str] = None


class SimpleWorkingMemory:
    """
    Простая реализация рабочей памяти (замена отсутствующего WorkingMemory).
    """
    
    def __init__(self, max_size: int = 100):
        self.messages = []
        self.max_size = max_size
    
    def add_message(self, role: str, content: str):
        """Добавление сообщения в память."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Ограничиваем размер памяти
        if len(self.messages) > self.max_size:
            self.messages = self.messages[-self.max_size:]
    
    def get_messages(self) -> List[Dict]:
        """Получение всех сообщений."""
        return self.messages
    
    def clear(self):
        """Очистка памяти."""
        self.messages = []
    
    def get_last_n(self, n: int) -> List[Dict]:
        """Получение последних n сообщений."""
        return self.messages[-n:] if n > 0 else []


class SimpleGuardrails:
    """
    Простая реализация guardrails (замена отсутствующего InputGuardrails).
    """
    
    def validate_input(self, query: str) -> Any:
        """Проверка входного запроса."""
        class SecurityCheck:
            def __init__(self, is_safe, reason=""):
                self.is_safe = is_safe
                self.reason = reason
        
        # Простая проверка на опасные паттерны
        dangerous_patterns = [
            "drop table", "delete from", "rm -rf", 
            "format c:", "exec(", "eval("
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if pattern in query_lower:
                return SecurityCheck(False, f"Обнаружен опасный паттерн: {pattern}")
        
        return SecurityCheck(True, "")


class AIAgent:
    """
    Основной класс AI-агента для интеллектуальной системы управления видеокамерой.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Инициализация AI-агента.
        """
        self.config = config or AgentConfig()
        logger.info(f"Инициализация агента: {self.config.name} v{self.config.version}")

        # Инициализация LLM
        self.llm = self._init_llm()
        logger.info("LLM инициализирован")

        # Инициализация инструментов
        self.tools = self._init_tools()
        logger.info(f"Доступно инструментов: {len(self.tools)}")

        # Инициализация памяти
        self.working_memory = None
        self.semantic_memory = None
        if self.config.memory_enabled:
            self.working_memory = SimpleWorkingMemory()
            if SEMANTIC_MEMORY_AVAILABLE:
                try:
                    self.semantic_memory = SemanticMemory()
                    logger.info("Семантическая память активирована")
                except Exception as e:
                    logger.error(f"Ошибка инициализации семантической памяти: {e}")
            else:
                logger.warning("Семантическая память не доступна")
            logger.info("Рабочая память активирована")

        # Инициализация guardrails
        self.guardrails = None
        if self.config.guardrails_enabled:
            self.guardrails = SimpleGuardrails()
            logger.info("Guardrails активированы")

        # Инициализация LangChain-агента
        self.agent = self._init_agent()
        logger.info("Агент готов к работе")

        # Статистика
        self.request_count = 0
        self.total_tokens = 0

    def _init_llm(self):
        """
        Инициализация LLM-клиента (YandexGPT).
        """
        iam_token = os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YANDEX_FOLDER_ID")

        if not iam_token or not folder_id:
            logger.warning(
                "Не найдены YANDEX_IAM_TOKEN или YANDEX_FOLDER_ID. "
                "Используется имитация LLM для тестирования."
            )
            from langchain_community.llms import FakeListLLM
            responses = [
                "Я проанализировал запрос. Для управления PTZ-камерой необходимо определить тип объекта и его координаты.",
                "Результат слежения: камера настроена, объект в центре кадра.",
                "Поиск завершён. Информация найдена."
            ]
            return FakeListLLM(responses=responses)

        try:
            from langchain_community.llms import YandexGPT
            return YandexGPT(
                iam_token=iam_token,
                folder_id=folder_id,
                temperature=self.config.temperature,
                max_tokens=1000
            )
        except ImportError:
            logger.error("LangChain Community не установлена")
            raise

    def _init_tools(self) -> List[Tool]:
        """
        Инициализация инструментов агента с правильной обёрткой для LangChain.
        """
        tools = []
        
        # Инструмент поиска
        try:
            search_tool = SearchTool()
            tools.append(Tool(
                name=search_tool.name,
                description=search_tool.description,
                func=search_tool._run
            ))
            logger.info("Зарегистрирован инструмент: SearchTool")
        except Exception as e:
            logger.error(f"Ошибка регистрации SearchTool: {e}")
        
        # Инструмент калькулятора
        try:
            calc_tool = CalculateTool()
            tools.append(Tool(
                name=calc_tool.name,
                description=calc_tool.description,
                func=calc_tool._run
            ))
            logger.info("Зарегистрирован инструмент: CalculateTool")
        except Exception as e:
            logger.error(f"Ошибка регистрации CalculateTool: {e}")
        
        # Инструмент управления PTZ-камерой
        try:
            custom_tool = CustomTool()
            tools.append(Tool(
                name=custom_tool.name,
                description=custom_tool.description,
                func=custom_tool._run
            ))
            logger.info("Зарегистрирован инструмент: CustomTool (PTZ Camera Tracker)")
        except Exception as e:
            logger.error(f"Ошибка регистрации CustomTool: {e}")
        
        return tools

    def _init_agent(self):
        """
        Инициализация LangChain-агента с использованием AgentExecutor.
        """
        from langchain.agents import create_react_agent
        from langchain.prompts import PromptTemplate
        
        # Создаем промпт для ReAct агента
        template = """Ты - интеллектуальный помощник для управления PTZ-камерой с функцией слежения.

У тебя есть доступ к следующим инструментам:

{tools}

Используй следующий формат:

Question: вопрос пользователя
Thought: подумай, что нужно сделать
Action: название инструмента из [{tool_names}]
Action Input: входные данные для инструмента
Observation: результат работы инструмента
... (повторяй Thought/Action/Action Input/Observation N раз)
Thought: Я знаю ответ
Final Answer: ответ пользователю

Начинай!

Question: {input}
Thought: {agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Создаем агента
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Создаем исполнителя агента
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True
        )
        
        return agent_executor

    def run(self, query: str, session_id: Optional[str] = None) -> AgentResponse:
        """
        Выполнение запроса к агенту.
        """
        start_time = time.time()
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"📨 Новый запрос: {query[:100]}...")

        # Проверка безопасности входа
        if self.guardrails:
            try:
                security_check = self.guardrails.validate_input(query)
                if not security_check.is_safe:
                    logger.warning(f"⛔ Запрос отклонён guardrails: {security_check.reason}")
                    return AgentResponse(
                        success=False,
                        answer="Запрос отклонён системой безопасности.",
                        steps=[],
                        duration_ms=0,
                        tokens_used=0,
                        error=security_check.reason
                    )
            except Exception as e:
                logger.error(f"Ошибка при проверке guardrails: {e}")

        try:
            # Выполнение запроса через агента
            logger.info("🔄 Выполнение запроса через агента...")
            result = self.agent.invoke({"input": query})
            answer = result.get("output", "Нет ответа")

            # Сохранение в рабочую память
            if self.working_memory:
                try:
                    self.working_memory.add_message("user", query)
                    self.working_memory.add_message("assistant", answer)
                    logger.debug("Сохранено в рабочую память")
                except Exception as e:
                    logger.error(f"Ошибка сохранения в рабочую память: {e}")

            # Сохранение в семантическую память
            if self.semantic_memory and session_id:
                try:
                    self.semantic_memory.add_document(
                        content=f"Query: {query}\nAnswer: {answer}",
                        metadata={
                            "session_id": session_id,
                            "type": "interaction",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    logger.debug("Сохранено в семантическую память")
                except Exception as e:
                    logger.error(f"Ошибка сохранения в семантическую память: {e}")

            # Формирование ответа
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)

            self.request_count += 1
            tokens_used = self._estimate_tokens(query, answer)
            self.total_tokens += tokens_used

            response = AgentResponse(
                success=True,
                answer=answer,
                steps=[],
                duration_ms=duration_ms,
                tokens_used=tokens_used
            )

            logger.info(f"✅ Запрос выполнен за {duration_ms}мс, токенов: {tokens_used}")
            return response

        except Exception as e:
            logger.error(f"❌ Ошибка выполнения: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                answer="Произошла ошибка при обработке запроса.",
                steps=[],
                duration_ms=0,
                tokens_used=0,
                error=str(e)
            )

    def _estimate_tokens(self, input_text: str, output_text: str) -> int:
        """Оценка количества использованных токенов."""
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4
        tool_overhead = 50
        return input_tokens + output_tokens + tool_overhead

    def get_stats(self) -> Dict:
        """Получение статистики агента."""
        stats = {
            "name": self.config.name,
            "version": self.config.version,
            "tools_count": len(self.tools),
            "tools_names": [tool.name for tool in self.tools],
            "memory_enabled": self.config.memory_enabled,
            "guardrails_enabled": self.config.guardrails_enabled,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens
        }
        
        if self.semantic_memory:
            try:
                stats["semantic_memory"] = self.semantic_memory.get_stats()
            except Exception as e:
                logger.error(f"Ошибка получения статистики памяти: {e}")
        
        return stats

    def clear_memory(self) -> None:
        """Очистка рабочей памяти агента."""
        if self.working_memory:
            try:
                self.working_memory.clear()
                logger.info("Рабочая память очищена")
            except Exception as e:
                logger.error(f"Ошибка очистки памяти: {e}")


# Точка входа для тестирования
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("Интеллектуальная система управления видеокамерой с функцией слежения за объектами")
    print("Автор: Ходжиев Фируз Фарходович")
    print("Специальность: Математическое обеспечение и администрирование ИС")
    print("=" * 80)

    # Создание агента
    print("\n🔄 Инициализация агента...")
    agent = AIAgent()

    # Вывод статистики
    print("\n📊 Статистика агента:")
    stats = agent.get_stats()
    for key, value in stats.items():
        if key == "tools_names":
            print(f"   {key}: {', '.join(value)}")
        else:
            print(f"   {key}: {value}")

    # Тестовые запросы
    test_queries = [
        "Рассчитай результат выражения (1500 + 1750) / 2",
        "Что такое PTZ-камера?",
    ]

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ АГЕНТА")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Тест {i}: {query}")
        print("-" * 50)
        
        response = agent.run(query)
        
        print(f"\n🤖 Ответ агента:")
        print(response.answer[:500])
        print(f"\n⏱️ Время: {response.duration_ms}мс | Токены: {response.tokens_used}")
        
        if response.error:
            print(f"❌ Ошибка: {response.error}")
        
        print("-" * 50)

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)