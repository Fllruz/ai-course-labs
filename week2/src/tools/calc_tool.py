# -*- coding: utf-8 -*-
"""
Инструмент калькулятора для математических вычислений
Лабораторная работа №2
Автор: Ходжиев Фируз Фарходович
Специальность: Математическое обеспечение и администрирование информационных систем
Тема диплома: Интеллектуальная система управления видеокамерой с функцией слежения за объектами
"""
from langchain.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import logging
import ast
import operator

logger = logging.getLogger(__name__)


class CalculatorInput(BaseModel):
    """Схема входных параметров для калькулятора."""
    expression: str = Field(
        description="Математическое выражение для вычисления (например: '2 + 2', '15 * 4', '(10+5)*3')",
        min_length=1,
        max_length=200
    )


class CalculateTool(BaseTool):
    """
    Инструмент для выполнения математических вычислений.

    Назначение:
    Вычисление математических выражений, поддержка базовых операций.

    Поддерживаемые операции:
    • Сложение (+)
    • Вычитание (-)
    • Умножение (*)
    • Деление (/)
    • Степень (**)
    • Скобки для группировки

    Интеграция с дипломом:
    Используется для расчётов углов поворота камеры, коэффициентов зума,
    преобразования координат и других математических операций в системе
    управления видеокамерой.
    """

    name = "calculator"
    description = """
    Выполняет математические вычисления. Поддерживает операции: +, -, *, /, ** и скобки.
    Входные данные: математическое выражение (например: '15 * 4', '(10+5)*3', '2**3').
    Возвращает: результат вычисления.
    """
    args_schema: Type[BaseModel] = CalculatorInput

    # Безопасные операции для eval
    _ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }

    def _run(self, expression: str) -> str:
        """
        Выполнение математического вычисления.

        Args:
            expression: Математическое выражение

        Returns:
            str: Результат вычисления
        """
        logger.info(f"Вычисление: {expression}")

        # Очистка выражения
        expression = expression.strip().replace(' ', '')

        # Проверка на пустое выражение
        if not expression:
            return "Ошибка: пустое выражение"

        try:
            # Безопасное вычисление через ast
            result = self._safe_eval(expression)
            
            # Форматирование результата
            if isinstance(result, float):
                # Округление до 5 знаков для читаемости
                result = round(result, 5)
                # Убираем .0 для целых чисел
                if result.is_integer():
                    result = int(result)
            
            logger.info(f"Результат: {result}")
            return f"Результат вычисления {expression} = {result}"
            
        except ZeroDivisionError:
            return "Ошибка: деление на ноль"
        except SyntaxError as e:
            return f"Ошибка синтаксиса: {e}"
        except Exception as e:
            return f"Ошибка вычисления: {e}"

    def _safe_eval(self, expression: str) -> float:
        """
        Безопасное вычисление математического выражения с использованием AST.

        Args:
            expression: Математическое выражение

        Returns:
            float: Результат вычисления
        """
        # Парсинг выражения в AST
        tree = ast.parse(expression, mode='eval')
        
        # Функция для безопасного вычисления узлов AST
        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op = type(node.op)
                if op in self._ALLOWED_OPS:
                    return self._ALLOWED_OPS[op](left, right)
                else:
                    raise ValueError(f"Неподдерживаемая операция: {op}")
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op = type(node.op)
                if op in self._ALLOWED_OPS:
                    return self._ALLOWED_OPS[op](operand)
                else:
                    raise ValueError(f"Неподдерживаемая унарная операция: {op}")
            else:
                raise ValueError(f"Неподдерживаемый тип узла: {type(node)}")
        
        return _eval(tree.body)

    async def _arun(self, expression: str) -> str:
        """Асинхронная версия."""
        return self._run(expression)

    def to_langchain_tool(self) -> BaseTool:
        """Конвертация в формат LangChain."""
        return self


# Для обратной совместимости (если кто-то импортирует CalculatorTool)
CalculatorTool = CalculateTool


# Точка входа для тестирования
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    tool = CalculateTool()
    
    print("=" * 60)
    print("Тестирование калькулятора для системы видеослежения")
    print("=" * 60)
    
    test_expressions = [
        "15 + 25",
        "15 + 25 * 4",
        "(15 + 25) * 4",
        "100 / 4",
        "2 ** 8",
        "1920 / 2",
        "1080 * 0.05",
        "180 / 1.5",
        "(1500 + 1750) / 2"
    ]
    
    for expr in test_expressions:
        print(f"\n📐 Выражение: {expr}")
        result = tool.run(expr)
        print(f"   {result}")
    
    print("\n" + "=" * 60)