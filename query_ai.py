from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.schema import CreateTable
from sqlalchemy.engine import Engine
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Step 1: Connect to your SQLite database
engine: Engine = create_engine(r"sqlite:///C:\Users\ranranjith\prem1b-project-2\sampledb.db")

metadata = MetaData()
metadata.reflect(bind=engine)

ddl_statements = []
for table in metadata.sorted_tables:
    ddl = CreateTable(table).compile(engine, compile_kwargs={"literal_binds": True})
    ddl_str = str(ddl).rstrip().rstrip(";")
    ddl_statements.append(ddl_str + ";")

schema_str = "\n\n".join(ddl_statements)

# Step 2: Prepare prompt with examples
examples = """
# -- EXAMPLES:
# -- Question: List all students and their schools
# -- SQL: SELECT students.name, schools.school_name FROM students JOIN schools ON students.school_id = schools.id;

# -- Question: List schools opened after 2010
# -- SQL: SELECT school_name FROM schools WHERE year_opened > 2010;

# -- Question: List all students above age 18
# -- SQL: SELECT name FROM students WHERE age > 18;
"""

prompt_template = f"""
-- You are an AI that converts questions into SQL queries using this schema:

{schema_str}

{examples}

-- Now answer this:
-- Question: {{your_question}}
-- SQL:
"""

# Step 3: Load model
model_name = "premai-io/prem-1B-SQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 4: Get user input
question = input("Ask your question: ").strip()

# Step 5: Generate SQL
prompt = prompt_template.replace("{your_question}", question).strip()
prompt += f"\n-- Question: {question}\n-- SQL:"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
sql_query = decoded_output.split("-- SQL:")[-1].strip().split("\n")[0]

print("\n✅ Generated SQL Query:\n", sql_query)

# ✅ Step 6: Execute SQL query and print results
try:
    with engine.connect() as connection:
        result = connection.execute(text(sql_query))
        rows = result.fetchall()

        if rows:
            print("\n📊 Query Results:")
            for row in rows:
                print(row)
        else:
            print("\n⚠️ No data returned.")

except Exception as e:
    print("\n❌ Error while executing the query:", e)
