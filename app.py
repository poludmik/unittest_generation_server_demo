from flask import Flask, jsonify, request
from langchain_openai import OpenAI

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


def call_gpt(users_code_chunk: str = ""):
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
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


@app.route('/generate_unittest', methods=['GET'])
def generate_unittest():
    try:
        received_data = request.get_json()

        if 'code_chunk' not in received_data:
            return jsonify({'error': 'Missing \'code_chunk\' in the JSON body of the request'}), 400
        
        return call_gpt(received_data['code_chunk'])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
