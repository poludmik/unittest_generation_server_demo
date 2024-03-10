from flask import Flask, jsonify, request
from langchain_openai import OpenAI
from transformers import AutoTokenizer
import transformers
import torch


BLUE = "\033[34m"
END = "\033[0m"


app = Flask(__name__)


TEST_EXAMPLE = """For the following code:
```python
def simulate_trading_strategy(prices):
    short_term_period = 5
    long_term_period = 20
    short_sma = [0] * len(prices)
    long_sma = [0] * len(prices)
    for i in range(len(prices)):
        if i >= short_term_period - 1:
            short_sma[i] = sum(prices[i-short_term_period+1:i+1]) / short_term_period
        if i >= long_term_period - 1:
            long_sma[i] = sum(prices[i-long_term_period+1:i+1]) / long_term_period
    positions = [0] * len(prices)
    for i in range(1, len(prices)):
        if short_sma[i] > long_sma[i] and short_sma[i-1] <= long_sma[i-1]:
            positions[i] = 1  # Buy signal
        elif short_sma[i] < long_sma[i] and short_sma[i-1] >= long_sma[i-1]:
            positions[i] = -1  # Sell signal
    cash = 10000  # Starting cash
    shares = 0
    for i in range(len(prices)):
        if positions[i] == 1:  # Buy
            shares_bought = cash // prices[i]
            cash -= shares_bought * prices[i]
            shares += shares_bought
        elif positions[i] == -1 and shares > 0:  # Sell
            cash += shares * prices[i]
            shares = 0
    final_value = cash + shares * prices[-1]
    return final_value - 10000  # Profit or loss
```
The unit tests are:
```python
class TestTradingStrategySimulation(unittest.TestCase):
    def test_profit_scenario(self):
        # Define a price scenario where the strategy should result in a profit
        prices = [100, 105, 102, 110, 108, 112, 115, 110, 120, 125]
        result = simulate_trading_strategy(prices)
        # Check if the profit (result) is greater than zero
        self.assertGreater(result, 0, "The strategy should yield a profit in this scenario.")

    def test_loss_scenario(self):
        # Define a price scenario where the strategy should result in a loss
        prices = [125, 120, 118, 117, 115, 114, 113, 112, 110, 108]
        result = simulate_trading_strategy(prices)
        # Check if the loss (result) is less than zero
        self.assertLess(result, 0, "The strategy should yield a loss in this scenario.")
```
"""

PROMPT = """So I have a piece of code that I want to test. Here it is:
```python
{users_code_chunk}
```

I want to write a unit test for this code. Can you help me write a unit tests function for this code?
Here is an example of unit tests that were written by an expert:
{one_shot_example}

Follow the backticks notation and output only the code for the unit tests class that uses unittest.TestCase.
"""


def call_gpt(users_code_chunk: str = "", model_name: str = "gpt-3.5-turbo-instruct"):
    print(f"Using {BLUE}'{model_name}'{END} LLM.")
    llm = OpenAI(model_name=model_name)
    question = PROMPT.format(users_code_chunk=users_code_chunk, one_shot_example=TEST_EXAMPLE)
    answer = llm.invoke(
                        question,
                        max_tokens=1000, 
                        temperature=0, 
                        top_p=1.0, 
                        frequency_penalty=0, 
                        presence_penalty=0
                        )
    assert type(answer) == str
    return answer


def call_codellama(users_code_chunk: str = "", model_name: str = "codellama/CodeLlama-7b-Instruct-hf"):
    print(f"Using {BLUE}'{model_name}'{END} LLM.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model="codellama/CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    question = PROMPT.format(users_code_chunk=users_code_chunk, one_shot_example=TEST_EXAMPLE)

    sequences = pipeline(
        question,
        do_sample=False,
        temperature=1e-9,
        top_p=1.0,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1000,
    )
    
    assert type(sequences[0]['generated_text']) == str

    return sequences[0]['generated_text']


@app.route('/generate_unittest', methods=['GET'])
def generate_unittest():
    try:
        received_data = request.get_json()

        if 'code_chunk' not in received_data:
            return jsonify({'error': 'Missing \'code_chunk\' in the JSON body of the request'}), 400
        
        if 'model_name' not in received_data:
            return call_gpt(received_data['code_chunk'])
        else:
            if received_data["model_name"].startswith("gpt"):
                return call_gpt(received_data['code_chunk'], model_name=received_data['model_name'])
            
            if received_data['model_name'].startswith("codellama"):
                return call_codellama(received_data['code_chunk'], model_name=received_data["model_name"])
        
        return 'Invalid model name: either gpt-... or codellama... are supported.'

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
