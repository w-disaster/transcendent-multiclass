FROM python:3.11

WORKDIR /usr/app

RUN pip install poetry

COPY . .

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "ice.py", "-k 10", "-n 10", "--pval-consider", "full-train"]