To setup application:

1. pipenv shell
2. pipenv install
3. python -m pip install langchain_google_community
4. python -m pip install langchain_google_vertexai

Authenticate with Google Cloud 
1. gcloud auth application-default login
2. gcloud auth application-default set-quota-project <project-id>

To run the application: 

```
streamlit run main.py
```
