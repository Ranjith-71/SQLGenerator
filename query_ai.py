

from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3

# Load model and tokenizer
model_name = "premai-io/prem-1B-SQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 💡 Enhanced schema + examples to improve model accuracy
schema = """
-- You are an AI that converts questions into SQL queries using this schema:

CREATE TABLE students (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    school_id INTEGER
);

CREATE TABLE schools (
    id INTEGER PRIMARY KEY,
    school_name TEXT,
    year_opened INTEGER
);

CREATE TABLE courses (
    id INTEGER PRIMARY KEY,
    course_name TEXT,
    school_id INTEGER
);

CREATE TABLE enrollments (
    student_id INTEGER,
    course_id INTEGER,
    enrollment_date DATE,
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(id),
    FOREIGN KEY (course_id) REFERENCES courses(id)
);

CREATE TABLE instructors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    course_id INTEGER,
    FOREIGN KEY (course_id) REFERENCES courses(id)
);

-- EXAMPLES:
-- Question: List all students and their schools
-- SQL: SELECT students.name, schools.school_name FROM students JOIN schools ON students.school_id = schools.id;

-- Question: List schools opened after 2010
-- SQL: SELECT school_name FROM schools WHERE year_opened > 2010;

-- Question: List all students above age 18
-- SQL: SELECT name FROM students WHERE age > 18;

-- Question: List all courses offered by a specific school
-- SQL: SELECT course_name FROM courses WHERE school_id = (SELECT id FROM schools WHERE school_name = 'Specific School');

-- Question: List all students enrolled in a specific course
-- SQL: SELECT students.name FROM students JOIN enrollments ON students.id = enrollments.student_id JOIN courses ON enrollments.course_id = courses.id WHERE courses.course_name = 'Specific Course';

-- Question: List all instructors teaching a specific course
-- SQL: SELECT instructors.name FROM instructors JOIN courses ON instructors.course_id = courses.id WHERE courses.course_name = 'Specific Course';

-- Now answer this:
-- Question: {your_question}
-- SQL:
"""


# 👇 Ask the user
question = input("Ask your question: ").strip()
prompt = schema.replace("{your_question}", question)

# Tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate SQL (no randomness for accuracy)
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and extract SQL only
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
sql_query = decoded_output.split("-- SQL:")[-1].strip().split("\n")[0]

print("\n✅ Generated SQL Query:\n", sql_query)

