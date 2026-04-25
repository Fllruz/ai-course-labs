# -*- coding: utf-8 -*-
"""
Специализированный RAG для строительной отрасли
Лабораторная работа №5
Автор: Ходжиев Фируз Фарходович
Специальность: Строительство
Тема диплома: Интеллектуальная система управления видеокамерой с функцией слежения за объектами
"""
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
import time
from langchain.prompts import ChatPromptTemplate
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class ConstructionRAGPipeline:
    """
    RAG-система для вопросов по строительству и системам видеонаблюдения.
    
    Особенности:
    • Приоритет нормативным документам (СНиП, ГОСТ, СП)
    • Точное цитирование требований безопасности
    • Классификация по типам документации
    • Поддержка вопросов по видеонаблюдению и слежению за объектами
    """
    
    def __init__(self, vectorstore, llm=None, top_k: int = 5):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        
        # Специфичный промпт для строительства
        self.rag_prompt = ChatPromptTemplate.from_template("""
Ты — эксперт по строительству и системам видеонаблюдения.
Используй ТОЛЬКО предоставленный контекст для ответа.
Цитируй конкретные пункты нормативных документов (СНиП, ГОСТ, СП).

Если вопрос касается:
• Техники безопасности - укажи конкретный СНиП и пункт
• Монтажа оборудования - опиши требования к установке
• Видеонаблюдения - сошлись на ГОСТ Р 57774-2017
• Слежения за объектами - опиши технические требования

Контекст из документов:
{context}

Вопрос: {question}

Ответ:
""")
        
        logger.info("ConstructionRAGPipeline инициализирован")
    
    def _format_context(self, documents: List[Dict]) -> str:
        """Форматирование контекста из найденных документов."""
        formatted = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            content = doc.get('content', '')
            formatted.append(f"[Документ {i}: {source}]\n{content}")
        
        return "\n\n".join(formatted)
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Выполнение запроса к RAG системе."""
        start_time = time.time()
        
        logger.info(f"Строительный RAG запрос: {question[:100]}...")
        
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
                if hasattr(self.llm, 'generate'):
                    response = self.llm.generate([prompt])
                    
                    if isinstance(response, dict):
                        answer = response.get("text", "")
                    elif hasattr(response, 'generations'):
                        answer = response.generations[0][0].text
                    else:
                        answer = str(response)
                else:
                    response = self.llm.invoke(prompt)
                    answer = response if isinstance(response, str) else str(response)
            except Exception as e:
                logger.error(f"Ошибка генерации: {e}")
                answer = f"Ошибка генерации ответа: {e}"
        else:
            # Fallback без LLM
            answer = f"Найдено {len(search_results)} релевантных документов.\n\n"
            for i, doc in enumerate(search_results, 1):
                chunk_preview = doc['content'][:300]
                answer += f"{i}. {chunk_preview}...\n\n"
        
        # Шаг 4: Формирование результата
        execution_time = time.time() - start_time
        
        result = {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": search_results if include_sources else [],
            "sources_count": len(search_results),
            "execution_time": round(execution_time, 3),
            "timestamp": datetime.now().isoformat(),
            "domain": "construction_and_video_surveillance",
            "doc_types": self._classify_document_types(search_results)
        }
        
        logger.info(f"Ответ сгенерирован за {execution_time:.3f}с")
        
        return result
    
    def _classify_document_types(self, documents: List[Dict]) -> Dict[str, int]:
        """Классификация найденных документов по типам."""
        types = {
            "snip": 0,
            "gost": 0,
            "technical": 0,
            "safety": 0
        }
        
        for doc in documents:
            source = doc.get('metadata', {}).get('source', '').lower()
            content = doc.get('content', '').lower()
            
            if 'снип' in source or 'снип' in content:
                types["snip"] += 1
            elif 'гост' in source or 'гост' in content:
                types["gost"] += 1
            elif 'технический' in source or 'камера' in content or 'видео' in content:
                types["technical"] += 1
            elif 'безопасность' in source or 'безопасность' in content:
                types["safety"] += 1
        
        return types


# Тестирование
if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("=" * 80)
    print("СПЕЦИАЛИЗИРОВАННЫЙ RAG ДЛЯ СТРОИТЕЛЬСТВА И ВИДЕОНАБЛЮДЕНИЯ")
    print(f"Автор: Ходжиев Фируз Фарходович")
    print(f"Специальность: Строительство")
    print(f"Тема диплома: Интеллектуальная система управления видеокамерой с функцией слежения за объектами")
    print("=" * 80)
    
    # Инициализация векторного хранилища
    vectorstore = VectorStoreManager(
        persist_directory="./data/chroma_db",
        collection_name="construction_documents"
    )
    
    # Инициализация LLM
    llm = None
    try:
        from langchain_community.llms import YandexGPT
        
        iam_token = os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        
        if iam_token and folder_id:
            llm = YandexGPT(
                iam_token=iam_token,
                folder_id=folder_id,
                temperature=0.3,
                max_tokens=500
            )
            print("✅ YandexGPT подключён")
        else:
            print("⚠️ YandexGPT не подключён (токены не найдены)")
    except ImportError as e:
        print(f"⚠️ Ошибка импорта YandexGPT: {e}")
    
    # Создание строительного RAG
    construction_rag = ConstructionRAGPipeline(
        vectorstore=vectorstore,
        llm=llm,
        top_k=5
    )
    
    # Тестовые вопросы по строительству и видеонаблюдению
    test_questions = [
    "Согласно СНиП 12-03-2001, каковы требования к высоте ограждения строительной площадки и оборудованию его защитным козырьком в местах массового прохода людей?",
    "Какие правила безопасности при производстве работ грузоподъемными кранами устанавливает СНиП 12-03-2001?",
    "В каком случае, согласно СНиП 12-03-2001, переход монтажников по строительным конструкциям (балкам, фермам) запрещен и что необходимо использовать в качестве альтернативы?",
    "Какие требования СНиП 12-03-2001 предъявляются к организации освещения рабочих мест в темное время суток?",
    "Каковы общие требования СНиП 12-03-2001 к подготовке и содержанию производственных территорий и проходов к рабочим местам?"
]
    
    print("\n" + "=" * 80)
    print("ТЕСТОВЫЕ ЗАПРОСЫ ПО СТРОИТЕЛЬСТВУ И ВИДЕОНАБЛЮДЕНИЮ")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"ВОПРОС {i}: {question}")
        print(f"{'='*60}")
        
        result = construction_rag.query(question)
        
        if result["success"]:
            print(f"\nОТВЕТ:")
            print(result["answer"])
            
            print(f"\nМетаданные:")
            print(f" • Домен: {result.get('domain', 'N/A')}")
            print(f" • Типы документов: {result.get('doc_types', {})}")
            print(f" • Найдено источников: {result['sources_count']}")
            print(f" • Время выполнения: {result['execution_time']}с")
            
            if result["sources"]:
                print(f"\nИсточники:")
                for j, source in enumerate(result["sources"][:3], 1):
                    metadata = source.get('metadata', {})
                    print(f"  {j}. {metadata.get('source', 'Unknown')} "
                          f"(score: {source['similarity_score']:.3f})")
        else:
            print(f"\nОШИБКА: {result.get('answer', 'Неизвестная ошибка')}")
    
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)