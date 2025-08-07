from sqlalchemy import create_engine, MetaData
from sqlalchemy.schema import CreateTable
from sqlalchemy.engine import Engine

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Step 1: Connect to SQLite and extract schema
engine: Engine = create_engine(r"sqlite:///C:/Users/ranranjith/prem1b-project-2/sampledb.db")
metadata = MetaData()
metadata.reflect(bind=engine)

schema_str = "\n\n".join(
    str(CreateTable(t).compile(engine, compile_kwargs={"literal_binds": True})).strip(";") + ";"
    for t in metadata.sorted_tables
)

# Step 2: Better examples to guide the model
examples = """
# -- EXAMPLES:
# -- Question: List all students and their schools
# -- SQL: SELECT students.name, schools.school_name FROM students JOIN schools ON students.school_id = schools.id;

# -- Question: List schools opened after 2010
# -- SQL: SELECT school_name FROM schools WHERE year_opened > 2010;

# -- Question: Show course names and how many students are enrolled in each
# -- SQL: SELECT c.course_name, COUNT(e.student_id) FROM courses c JOIN enrollments e ON c.id = e.course_id GROUP BY c.course_name;

# -- Question: Find all students over 18 enrolled in courses taught by 'Dr. Strange'
# -- SQL: SELECT s.name FROM students s JOIN enrollments e ON s.id = e.student_id JOIN instructors i ON e.course_id = i.course_id WHERE s.age > 18 AND i.name = 'Dr. Strange';

# -- Question: List all employees earning more than 60000
# -- SQL: SELECT name FROM emp WHERE salary > 60000;

# -- Question: List all schools with more than 10 students
# -- SQL: SELECT school_name FROM schools WHERE id IN (SELECT school_id FROM students GROUP BY school_id HAVING COUNT(*) > 10);
"""

prompt_template = f"""
-- You are an AI that writes SQL queries from questions using this schema:

{schema_str}

{examples}

-- Now answer this:
-- Question: {{your_question}}
-- SQL:
"""

# Step 3: Load model/tokenizer only once
model_name = "premai-io/prem-1B-SQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 4: Get user question
question = input("Ask your question: ").strip()
prompt = prompt_template.replace("{your_question}", question)

# Step 5: Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

# Generate SQL (no randomness for accuracy)
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,  # deterministic output
    pad_token_id=tokenizer.eos_token_id
)

# Step 6: Decode and extract clean SQL
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
sql_query = decoded_output.split("-- SQL:")[-1].strip().split("\n")[0]

print("\n✅ Generated SQL Query:\n", sql_query)
