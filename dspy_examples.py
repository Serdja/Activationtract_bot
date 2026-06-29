"""
DSPy — примеры использования.

DSPy позволяет строить AI-пайплайны декларативно:
вместо написания промптов ты описываешь сигнатуры (вход -> выход),
а фреймворк сам генерирует и оптимизирует промпты.
"""

import dspy


# ============================================================
# 1. НАСТРОЙКА МОДЕЛИ (LM)
# ============================================================
# DSPy поддерживает OpenAI, Anthropic, локальные модели и др.
# Раскомментируй нужную строку и подставь свой API-ключ:

# OpenAI:
# lm = dspy.LM("openai/gpt-4o-mini", api_key="sk-...")

# Anthropic (Claude):
# lm = dspy.LM("anthropic/claude-sonnet-4-20250514", api_key="sk-ant-...")

# Локальная модель через Ollama:
# lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434")

# Для демо без ключа — покажем структуру кода:
print("=== DSPy Примеры ===\n")


# ============================================================
# 2. SIGNATURES — описание задачи
# ============================================================
# Самый простой способ — строка "вход -> выход":

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
# 3. MODULES — строительные блоки
# ============================================================

# Простой вызов (Predict) — прямой промпт
qa_simple = dspy.Predict(QA)

# Chain of Thought — модель рассуждает шаг за шагом
qa_cot = dspy.ChainOfThought(QA)

# Можно строить сложные пайплайны как классы:
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
# 4. ИСПОЛЬЗОВАНИЕ (нужен настроенный LM)
# ============================================================
def demo_with_lm():
    """Запусти эту функцию после настройки модели."""

    # Простой вопрос-ответ
    result = qa_simple(question="Что такое DSPy?")
    print(f"Простой ответ: {result.answer}")

    # Chain of Thought
    result = qa_cot(question="Почему небо голубое?")
    print(f"CoT рассуждение: {result.reasoning}")
    print(f"CoT ответ: {result.answer}")

    # RAG пайплайн
    rag = RAGPipeline()
    result = rag(
        question="Какой город столица Франции?",
        context="Франция — страна в Западной Европе. Столица — Париж.",
    )
    print(f"RAG ответ: {result.answer}")

    # Многошаговый анализ
    analyzer = MultiStepAnalyzer()
    result = analyzer(text="Этот продукт превосходен! Отличное качество и быстрая доставка.")
    print(f"Резюме: {result.summary}")
    print(f"Тональность: {result.sentiment}")


# ============================================================
# 5. ОПТИМИЗАЦИЯ (самая мощная фича DSPy)
# ============================================================
def demo_optimization():
    """
    Оптимизаторы автоматически подбирают лучшие промпты.
    Нужен датасет с примерами и метрика оценки.
    """

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


# ============================================================
# 6. БЫСТРЫЙ СТАРТ
# ============================================================
print("""
Как начать использовать DSPy:

1. Установи: pip install dspy
2. Настрой модель:
   import dspy
   lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_KEY")
   dspy.configure(lm=lm)

3. Опиши задачу через Signature:
   qa = dspy.Predict("question -> answer")
   result = qa(question="Что такое Python?")
   print(result.answer)

4. Используй Chain of Thought для сложных задач:
   qa = dspy.ChainOfThought("question -> answer")

5. Оптимизируй с помощью BootstrapFewShot:
   optimizer = dspy.BootstrapFewShot(metric=my_metric)
   optimized = optimizer.compile(qa, trainset=examples)

Доступные модули:
  - dspy.Predict        — простой вызов
  - dspy.ChainOfThought — рассуждение шаг за шагом
  - dspy.ReAct           — агент с инструментами
  - dspy.ProgramOfThought — генерация кода для вычислений
  - dspy.MultiChainComparison — сравнение нескольких цепочек

Доступные оптимизаторы:
  - dspy.BootstrapFewShot       — подбор few-shot примеров
  - dspy.MIPROv2                — оптимизация промптов и примеров
  - dspy.BootstrapFinetune      — файнтюнинг модели
""")


if __name__ == "__main__":
    print("Для запуска демо с реальной моделью:")
    print("1. Раскомментируй настройку LM в начале файла")
    print("2. Вызови demo_with_lm() или demo_optimization()")
