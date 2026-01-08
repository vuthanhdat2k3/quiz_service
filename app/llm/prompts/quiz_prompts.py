"""
Quiz generation prompt templates
Tất cả các prompt template để tạo câu hỏi quiz

CHÚ Ý: File này CHỈ chứa các prompt templates (string constants).
Không chứa code logic - logic nằm trong base_adapter.py
"""

# ============================================================================
# LANGUAGE INSTRUCTIONS - Hướng dẫn ngôn ngữ
# ============================================================================

LANGUAGE_INSTRUCTIONS = {
    "vi": "Generate the question, choices, and explanation in Vietnamese (Tiếng Việt).",
    "en": "Generate the question, choices, and explanation in English.",
}

LANGUAGE_INSTRUCTIONS_ALL = {
    "vi": "Generate ALL questions, choices, and explanations in Vietnamese (Tiếng Việt).",
    "en": "Generate ALL questions, choices, and explanations in English.",
}

LANGUAGE_INSTRUCTIONS_SHORT = {
    "vi": "Generate the question, answer, and explanation in Vietnamese (Tiếng Việt).",
    "en": "Generate the question, answer, and explanation in English.",
}

LANGUAGE_INSTRUCTIONS_TF = {
    "vi": "Generate the statement and explanation in Vietnamese (Tiếng Việt).",
    "en": "Generate the statement and explanation in English.",
}

# ============================================================================
# VARIETY HINTS - Gợi ý để tạo câu hỏi đa dạng
# ============================================================================

VARIETY_HINTS = [
    "Focus on definitions and key concepts.",
    "Focus on applications and examples.",
    "Focus on relationships and comparisons.",
    "Focus on causes and effects.",
    "Focus on processes and procedures.",
    "Focus on advantages and disadvantages.",
    "Focus on specific details and facts.",
]

# ============================================================================
# PROMPT TEMPLATES - Các template prompt
# ============================================================================

# Single MCQ Question Template
MCQ_PROMPT_TEMPLATE = """You are an exam question generator for academic content. Given the content below, {instruction} that tests understanding of the content.

{lang_instruction}

Question #{question_number} focus: {variety_hint}
{avoid_section}
Output only JSON with exactly these keys: {{"question":"", "choices":["","","",""], {answer_format}, "explanation":"", "difficulty":"easy|medium|hard"}}

Content:
"{passage}"

Output JSON:"""

# Batch MCQ Questions Template
BATCH_MCQ_PROMPT_TEMPLATE = """You are an expert exam question generator. Generate exactly {num_questions} UNIQUE multiple-choice questions based on the content below.

FOLLOW THIS BLUEPRINT (order matters, do not swap items):
{question_plan_text}

Type rules:
- single_choice → exactly ONE correct answer. Answer format: "answer":"A"
- multiple_choice → exactly TWO correct answers. Answer format: "answer":["A","B"]

Difficulty expectations:
{difficulty_summary}

Additional constraints:
- Each question must test a DIFFERENT concept or aspect
- No duplicate or near-duplicate questions
- Provide 4 concise, distinct options (A-D)

{lang_instruction}

Output ONLY a JSON array with exactly {num_questions} objects, each with these keys:
[
    {{"question":"...", "choices":["A)...","B)...","C)...","D)..."], "answer":..., "explanation":"...", "difficulty":"easy|medium|hard"}},
    ...
]

Content to generate questions from:
\"\"\"
{passage}
\"\"\"

Output JSON array (exactly {num_questions} questions):"""

# Distractor Refinement Template
DISTRACTOR_PROMPT_TEMPLATE = """Input JSON:
{{"passage":"{passage}", "correct":"{correct_answer}", "candidates":[{candidates_json}]}}

Task: Return exactly three distractors (strings) that are plausible but incorrect given the passage. Do not repeat the correct answer. Ensure distractors are not directly supported by the passage.

Output: JSON array: ["...","...","..."]"""

# Short Answer Question Template
SHORT_ANSWER_PROMPT_TEMPLATE = """Generate a short answer question from the following passage. The question should require a brief text response (1-3 sentences).

{lang_instruction}

Output only JSON: {{"question":"", "answer":"", "explanation":"", "difficulty":"easy|medium|hard"}}

Passage:
"{passage}"

Output JSON:"""

# True/False Question Template  
TRUE_FALSE_PROMPT_TEMPLATE = """Generate a true/false statement based on the following passage. The statement should test understanding of a key fact.

{lang_instruction}

Output only JSON: {{"statement":"", "answer":true|false, "explanation":"", "difficulty":"easy|medium|hard"}}

Passage:
"{passage}"

Output JSON:"""

# ============================================================================
# INSTRUCTION TEXTS - Các text hướng dẫn có thể tái sử dụng
# ============================================================================

# Question type instructions
SINGLE_CHOICE_INSTRUCTION = "generate ONE multiple-choice question with exactly ONE correct answer"
MULTIPLE_CHOICE_INSTRUCTION_TEMPLATE = "generate ONE multiple-choice question with exactly {num_correct} correct answers"

# Answer format instructions
SINGLE_ANSWER_FORMAT = '"answer":"A|B|C|D"'
MULTIPLE_ANSWER_FORMAT = '"answer":["A","B"]  // Array of correct answer letters'

# Type instructions for batch generation
TYPE_INSTRUCTION_SINGLE = "Each question must have exactly ONE correct answer."
TYPE_INSTRUCTION_MULTIPLE = "Each question must have exactly 2 correct answers."
TYPE_INSTRUCTION_MIX = """IMPORTANT - Question Type Pattern:
- Questions 1, 3, 5 (odd numbers): SINGLE correct answer (1 correct choice)
- Questions 2, 4 (even numbers): MULTIPLE correct answers (exactly 2 correct choices)
You MUST follow this alternating pattern strictly."""

# Answer formats for batch generation
ANSWER_FORMAT_SINGLE = '"answer":"A"  // Single correct answer letter'
ANSWER_FORMAT_MULTIPLE = '"answer":["A","B"]  // Array of 2 correct answer letters'
ANSWER_FORMAT_MIX = '''For ODD questions (single-choice): "answer":"A"
For EVEN questions (multiple-choice): "answer":["A","B"] (exactly 2 correct answers)'''

# Avoid duplication section template
AVOID_DUPLICATION_TEMPLATE = """
IMPORTANT: Do NOT create questions similar to these existing ones:
{existing_questions_list}

Create a UNIQUE and DIFFERENT question that tests a different aspect or concept from the topic.
"""
