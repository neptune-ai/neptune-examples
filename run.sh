find . -type d -wholename "*/docs/*" -delete
python ci/create_doc_examples.py
python ci/run_doc_examples.py