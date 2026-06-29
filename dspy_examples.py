"""
DSPy — примеры использования с Google Gemini.

DSPy позволяет строить AI-пайплайны декларативно:
вместо написания промптов ты описываешь сигнатуры (вход -> выход),
а фреймворк сам генерирует и оптимизирует промпты.
"""

import os

import dspy
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ОШИБКА: Установи GEMINI_API_KEY в файле .env")
    print("Скопируй .env.example в .env и вставь свой ключ")
    print("Получить ключ: https://aistudio.google.com/apikey")
    exit(1)

lm = dspy.LM("gemini/gemini-2.0-flash", api_key=GEMINI_API_KEY)
dspy.configure(lm=lm)

print("=== DSPy + Google Gemini ===\n")


# ============================================================
# SIGNATURES — описание задачи
# ============================================================

class QA(dspy.Signature):
    """Ответь на вопрос кратко и точно."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Summarize(dspy.Signature):
    """Сделай краткое резюме текста."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()


class Classify(dspy.Signature):
    """Определи тональность текста."""
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive, negative, или neutral")


# ============================================================
# MODULES — строительные блоки
# ============================================================

qa_simple = dspy.Predict(QA)
qa_cot = dspy.ChainOfThought(QA)


class RAGPipeline(dspy.Module):
    """Пример RAG-пайплайна: поиск + генерация ответа."""

    def __init__(self):
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question, context):
        return self.generate(context=context, question=question)


class MultiStepAnalyzer(dspy.Module):
    """Пример многошагового анализа."""

    def __init__(self):
        self.summarize = dspy.Predict(Summarize)
        self.classify = dspy.Predict(Classify)

    def forward(self, text):
        summary = self.summarize(text=text)
        sentiment = self.classify(text=summary.summary)
        return dspy.Prediction(
            summary=summary.summary,
            sentiment=sentiment.sentiment,
        )


# ============================================================
# ДЕМО
# ============================================================

if __name__ == "__main__":
    print("--- 1. Простой вопрос-ответ ---")
    result = qa_simple(question="Что такое Python?")
    print(f"Ответ: {result.answer}\n")

    print("--- 2. Chain of Thought (рассуждение) ---")
    result = qa_cot(question="Почему небо голубое?")
    print(f"Рассуждение: {result.reasoning}")
    print(f"Ответ: {result.answer}\n")

    print("--- 3. RAG пайплайн ---")
    rag = RAGPipeline()
    result = rag(
        question="Какой город столица Франции?",
        context="Франция — страна в Западной Европе. Столица — Париж. Население ~67 млн.",
    )
    print(f"Ответ: {result.answer}\n")

    print("--- 4. Многошаговый анализ ---")
    analyzer = MultiStepAnalyzer()
    result = analyzer(text="Этот продукт превосходен! Отличное качество и быстрая доставка.")
    print(f"Резюме: {result.summary}")
    print(f"Тональность: {result.sentiment}\n")

    print("--- 5. Оптимизация промптов ---")
    trainset = [
        dspy.Example(question="Столица Франции?", answer="Париж").with_inputs("question"),
        dspy.Example(question="Столица Японии?", answer="Токио").with_inputs("question"),
        dspy.Example(question="Столица Бразилии?", answer="Бразилиа").with_inputs("question"),
    ]

    def metric(example, prediction, trace=None):
        return example.answer.lower() in prediction.answer.lower()

    optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
    optimized_qa = optimizer.compile(dspy.Predict(QA), trainset=trainset)

    result = optimized_qa(question="Столица Германии?")
    print(f"Оптимизированный ответ: {result.answer}")

    print("\n=== Готово! DSPy работает с Gemini ===")
