**[DEMO]** Flask server generating unittests for a given chunk of code.

* Gets a chunk of code as a string.
* Returns generated unittests as a string.

Install dependencies:
```sh
pip install openai langchain requests Flask langchain_openai transformers accelerate
```

The `/generate_unittest` endpoint accepts a json body with `code_chunk` field and returns a string in `response.content`.

The app uses OPENAI_API_KEY that needs to be set as an environment variable before running `python app.py`:

```sh
export OPENAI_API_KEY='<YOUR_KEY>'
```

You can also specify the OpenAI LLM to be used with `model_name` field, or a codellama model for local inference. Codellama-7B needs around 20GB of ram to run. Haven't tested codellama yet.
