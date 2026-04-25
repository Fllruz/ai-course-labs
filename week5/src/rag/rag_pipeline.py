# -*- coding: utf-8 -*-
"""
Основный RAG пайплайн для генерации ответов
Лабораторная работа №5
Дисциплина: Искусственный интеллект
Автор: Ходжиев Фируз Фарходович
Группа: МОА-221
Дата: 2026
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG пайплайн для генерации ответов на вопросы.

    Архитектура:
    1. Получение вопроса от пользователя
    2. Поиск релевантных документов в векторной базе
    3. Формирование промпта с контекстом
    4. Генерация ответа через LLM (YandexGPT)
    5. Возврат ответа с цитированием источников

    Атрибуты:
    vectorstore: Векторное хранилище
    llm: LLM клиент
    top_k: Количество документов для поиска
    """

    def __init__(
            self,
            vectorstore: VectorStoreManager,
            llm=None,
            top_k: int = 5
    ):
        """
        Инициализация RAG пайплайна.

        Args:
            vectorstore: Векторное хранилище
            llm: LLM клиент (YandexGPT/GigaChat)
            top_k: Количество документов для поиска
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

        # RAG промпт
        self.rag_prompt = ChatPromptTemplate.from_template("""
Ты — полезный ассистент для ответов на вопросы по документации.
Используй ТОЛЬКО предоставленный контекст для ответа.
Если ответа нет в контексте, скажи "В предоставленных документах нет информации по этому вопросу".

Контекст из документов:
{context}

Вопрос: {question}

Ответ:
""")

        logger.info("RAGPipeline инициализирован")

    def _format_context(self, documents: List[Dict]) -> str:
        """
        Форматирование контекста из найденных документов.

        Args:
            documents: Список найденных документов

        Returns:
            str: Форматированный контекст
        """
        formatted = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            content = doc.get('content', '')
            formatted.append(f"[Источник {i}: {source}]\n{content}")

        return "\n\n".join(formatted)

    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Выполнение запроса к RAG системе.

        Args:
            question: Вопрос пользователя
            include_sources: Включать ли источники в ответ

        Returns:
            Dict: Ответ с метаданными
        """
        start_time = time.time()

        logger.info(f"RAG запрос: {question[:100]}...")

        # Шаг 1: Поиск релевантных документов
        search_results = self.vectorstore.search_with_scores(
            query=question,
            k=self.top_k
        )

        if not search_results:
            return {
                "success": False,
                "answer": "Не найдено релевантных документов в базе знаний.",
                "sources": [],
                "execution_time": 0
            }

        # Шаг 2: Формирование контекста
        context = self._format_context(search_results)

        # Шаг 3: Генерация ответа через LLM
        if self.llm:
            prompt = self.rag_prompt.format(
                context=context,
                question=question
            )

            try:
                # ИСПРАВЛЕНИЕ: передаем список строк, а не строку
                if hasattr(self.llm, 'generate'):
                    # Для YandexGPT и подобных
                    response = self.llm.generate([prompt])  # Оборачиваем в список
                    if isinstance(response, dict):
                        answer = response.get("text", "")
                    elif hasattr(response, 'generations'):
                        answer = response.generations[0][0].text
                    else:
                        answer = str(response)
                else:
                    # Для других типов LLM
                    response = self.llm.invoke(prompt)
                    answer = response if isinstance(response, str) else str(response)
            except Exception as e:
                logger.error(f"Ошибка генерации: {e}")
                answer = f"Ошибка генерации ответа: {e}"
        else:
            # Fallback без LLM
            answer = f"Найдено {len(search_results)} релевантных документов.\n\n"
            for i, doc in enumerate(search_results, 1):
                answer += f"{i}. {doc['content'][:300]}...\n\n"

        # Шаг 4: Формирование результата
        execution_time = time.time() - start_time

        result = {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": search_results if include_sources else [],
            "sources_count": len(search_results),
            "execution_time": round(execution_time, 3),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"RAG ответ сгенерирован за {execution_time:.3f}с")

        return result

    def query_batch(
            self,
            questions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Пакетная обработка запросов.

        Args:
            questions: Список вопросов

        Returns:
            List[Dict]: Список ответов
        """
        results = []

        for i, question in enumerate(questions, 1):
            logger.info(f"Обработка запроса {i}/{len(questions)}")
            result = self.query(question)
            results.append(result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика RAG пайплайна."""
        return {
            "top_k": self.top_k,
            "llm_enabled": self.llm is not None,
            "vectorstore_stats": self.vectorstore.get_statistics()
        }


# Точка входа для тестирования
if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_community.llms import YandexGPT

    load_dotenv()

    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №5")
    print("ТЕСТИРОВАНИЕ RAG PIPELINE")
    print("=" * 80)

    # Инициализация компонентов
    from rag.vector_store import VectorStoreManager

    vectorstore = VectorStoreManager(
        persist_directory="./data/chroma_db",
        collection_name="rag_documents"
    )

    # YandexGPT
    iam_token = os.getenv("YANDEX_IAM_TOKEN")
    folder_id = os.getenv("YANDEX_FOLDER_ID")

    llm = None
    if iam_token and folder_id:
        llm = YandexGPT(
            iam_token=iam_token,
            folder_id=folder_id,
            temperature=0.3,
            max_tokens=500
        )
        print("✅ YandexGPT подключён")
    else:
        print("⚠️ YandexGPT не подключён (работаем без LLM)")

    # RAG пайплайн
    rag = RAGPipeline(
        vectorstore=vectorstore,
        llm=llm,
        top_k=5
    )

    # Тестовые запросы
    test_questions = [
        "Что такое искусственный интеллект?",
        "Как работает RAG система?",
        "Какие есть векторные базы данных?"
    ]

    print("\n" + "=" * 80)
    print("ТЕСТОВЫЕ ЗАПРОСЫ")
    print("=" * 80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"ВОПРОС {i}: {question}")
        print(f"{'=' * 60}")

        result = rag.query(question)

        print(f"\nОТВЕТ:")
        print(result["answer"])

        print(f"\nМетаданные:")
        print(f" • Найдено источников: {result['sources_count']}")
        print(f" • Время выполнения: {result['execution_time']}с")

        if result["sources"]:
            print(f"\nИсточники:")
            for j, source in enumerate(result["sources"][:3], 1):
                print(f"  {j}. {source['metadata'].get('source', 'Unknown')} "
                      f"(score: {source['similarity_score']:.3f})")

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)