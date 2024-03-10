**[DEMO]** Flask server generating unittests for a given chunk of code.

* Gets a chunk of code as a string.
* Returns generated unittests as a string.

Install dependencies:
```sh
pip install openai langchain requests Flask langchain_openai
```

The `/generate_unittest` endpoint accepts a json body with `code_chunk` attribute and returns a string in `response.content`.

The app uses OPENAI_API_KEY that needs to be set as an environment variable before running `python app.py`:

```sh
export OPENAI_API_KEY='<YOUR_KEY>'
```
