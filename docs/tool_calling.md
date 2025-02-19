# Tool Calling

Some models support tool calling (function calling, or code interpreter). It would be much easier (and cool) to do tool calling in Python.
Demos of tool calling for these models are provided:

* [ChatGLM3](../scripts/tool_glm3.py), [GLM-4](../scripts/tool_glm4.py), [CodeGeeX4](../scripts/tool_codegeex4.py)
* [Mistral-Instruct-7B-v0.3](../scripts/tool_mistral.py)
* [QWen v1.5 & v2](../scripts/tool_qwen.py)
* [DeepSeek-Coder v2](../scripts/tool_deepseekcoder.py) (Note: function calling is *officially* unsupported.)
* [InternLM 2.5](../scripts//tool_internlm.py)
* [NuminaMath](../scripts/tool_numinamath.py)
* [LlaMA3-Groq-Tool-Use](../scripts/tool_groq.py)
* [LlaMA 3.1](../scripts/tool_llama3.1.py)
* [Watt-Tool](../scripts/tool_watt.py)

## Precondition

Build target [`libchatllm`](binding.md).

## Demo

### ChatGLM3/GLM-4

Since tools for ChatGLM3/GLM-4 are encoded in system prompt which results a very long prompt, causing a large delay before generating the first output.
Saving the session and reload it later can eliminate such delay. To save the session exactly after system prompt is evaluated:

```
python tool_glm4.py -i -m /path/to/your/model.bin --save_session 0 /path/to/session/file
```

Below is a demo with GLM-4 which loads a previously saved session:

```
python tool_glm4.py -i --temp 0 -m :glm-4 --load_session /path/to/session/file
    ________          __  __    __    __  ___
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by GLM-4,                      /_/   /_/
with 9399951360 (9.4B) parameters.

You  > weather in beijing
A.I. > [Use Tool]: get_weather


The current weather in Beijing is sunny with a temperature of 34 degrees Celsius and a humidity of 14%. The perceived temperature is 32 degrees Celsius. The observation time is 06:09 AM.
You  > how about jinan?
A.I. > [Use Tool]: get_weather


The current weather in Jinan is sunny with a temperature of 37 degrees Celsius and a humidity of 16%. The perceived temperature is 35 degrees Celsius. The observation time is 08:44 AM.
You  > which city is hotter?
A.I. >
Based on the weather information provided, Jinan is currently hotter than Beijing. The temperature in Jinan is 37 degrees Celsius, while the temperature in Beijing is 34 degrees Celsius.
```

### Mistral

Tools for Mistral are provided together with user inputs (similar to OpenAI GPT models). In this demo, tools are selected by the leading ":...:",
which contains keywords to filter tools.

Note that, in the second round, no tools are provided, and the model is able to use the tool given in the first round.

```
python tool_mistral.py -i -m :mistral
    ________          __  __    __    __  ___
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by Mistral,                    /_/   /_/
with 7248023552 (7.2B) parameters.

You  > :weather: What's the weather like in Beijing now?
A.I. > [Use Tools]: [{"name": "get_weather", "arguments": {"city_name": "Beijing"}}]

 The current weather in Beijing is clear. The temperature is 32 degrees Celsius, and it feels like 30 degrees Celsius. The humidity is 32%. The observation was made at 1:59 PM.
You  > How about Jinan?
A.I. > [Use Tools]: [{"name": "get_weather", "arguments": {"city_name": "Jinan"}}]

 The current weather in Jinan is clear. The temperature is 30 degrees Celsius, and it feels like 28 degrees Celsius. The humidity is 28%. The observation was made at 2:19 PM.
You  > which city is hotter?
A.I. >  The temperature in Beijing is 32 degrees Celsius, while the temperature in Jinan is 30 degrees Celsius. So, Beijing is hotter than Jinan.
```

### QWen

Tool calling with QWen v1.5 & v2 is implemented in Python.

#### v1.5 MoE

```
python tool_qwen.py -i -m :qwen1.5:moe
    ________          __  __    __    __  ___ (通义千问)
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by QWen2-MoE,                  /_/   /_/
with 14315784192 (2.7B effect.) parameters.

You  > weather in beijing
A.I. > [Use Tool]: get_weather

 The current weather in Beijing is sunny and the temperature is 33 degrees Celsius.
You  > how about jinan?
A.I. > [Use Tool]: get_weather

 The current weather in Jinan is partly cloudy and the temperature is 36 degrees Celsius.
You  > which city is hotter?
A.I. > [Use Tool]: get_weather

 The temperature in Beijing is currently 33 degrees Celsius, while in Jinan it is 36 degrees Celsius. So, Jinan is hotter.
```

#### v2

```
python tool_qwen.py -i -m :qwen2:1.5b
    ________          __  __    __    __  ___ (通义千问)
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by QWen2,                      /_/   /_/
with 1543714304 (1.5B) parameters.

You  > weather in beijing
A.I. > [Use Tool]: get_weather

The current weather in Beijing is Sunny, with a temperature of 33°C, a feels like temperature of 31°C, and a humidity of 36%. The weather observation time is 03:35 AM.
You  > how about jinan
A.I. > [Use Tool]: get_weather

The current weather in Jinan is Partly cloudy, with a temperature of 36°C, a feels like temperature of 36°C, and a humidity of 27%. The weather observation time is 05:22 AM.
You  > which city is hotter?
A.I. > Jinan is hotter than Beijing. Jinan's temperature is 36°C and Beijing's temperature is 33°C.
```

### DeepSeek-Coder v2

```
python tool_deepseekcoder.py -i -m :deepseek-coder-v2:light
    ________          __  __    __    __  ___
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by DeepSeek-V2,                /_/   /_/
with 15706484224 (2.7B effect.) parameters.

You  > weather in beijing?
A.I. > [Use Tool]: get_weather

北京当前的天气情况如下：
- 温度：24°C
- 体感温度：25°C
- 湿度：74%
- 天气描述：小雨
- 观测时间：06:47 AM
You  > how about jinan?
A.I. > [Use Tool]: get_weather

济南当前的天气情况如下：
- 温度：24°C
- 体感温度：26°C
- 湿度：80%
- 天气描述：局部有阵雨
- 观测时间：10:50 AM
```

### InternLM 2.5

```
python tool_internlm.py -i -m :internlm2.5 --temp 0
    ________          __  __    __    __  ___ (书生·浦语)
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by InternLM,                   /_/   /_/
with 7737708544 (7.7B) parameters.

You  > weather in beijing
A.I. > 好的，我需要调用一个API来获取北京的天气信息。请稍等片刻

[Use Tools]: get_weather
北京现在的天气是晴天，气温为30摄氏度，湿度为55%。有什么其他问题我可以帮您解答吗？
You  > how about jinan?
A.I. > 好的，我需要调用一个API来获取济南的天气信息。请稍等片刻

[Use Tools]: get_weather
济南现在的天气是晴天，气温为27摄氏度，湿度为58%。有什么其他问题我可以帮您解答吗？
You  > which city is hotter?
A.I. > 根据提供的天气信息，北京的温度为30摄氏度，而济南的温度为27摄氏度。因此，北京比济南更热。
```

### CodeGeeX4

```
python tool_codegeex4.py -m :codegeex4
    ________          __  __    __    __  ___
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by CodeGeeX4,                  /_/   /_/
with 9399951360 (9.4B) parameters.

You  >  what's the weather like in beijing?
A.I. >
[Use Tool] get_weather

The current weather in Beijing is sunny with a temperature of 29°C. The humidity is 58%. The observation time is 07:59 AM.
```

### NuminaMath

Note: F16 (`-m :numinamath:f16`) performs better (correct vs wrong) than Q8.

> ```
> python tool_numinamath.py --temp 0 -m :numinamath
>
> 💣💣 DANGEROUS! NO SAND-BOXING. DEMO ONLY. 💣💣
>     ________          __  __    __    __  ___
>    / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
>   / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
>  / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
>  \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
> You are served by NumiaMath,                  /_/   /_/
> with 6910365696 (6.9B) parameters.
> ```
> You  > show the result of 10!
>
> A.I. >
> Sure, let's calculate the factorial of 10 using Python. The factorial of a non-negative integer \( n \) (denoted as \( n! \)) is the product of all positive integers less than or equal to \( n \).
>
> For example, \( 10! = 10 \times 9 \times 8 \times 7 \times 6 \times 5 \times 4 \times 3 \times 2 \times 1 \).
>
> Let's compute \( 10! \) using Python.
> ```python
> import math
>
> # Calculate the factorial of 10
> result = math.factorial(10)
> print(result)
> ```
>
> [EXEC CODE]3628800
>
> The factorial of 10, denoted as \(10!\), is \(3,628,800\).
>
> Thus, the result of \(10!\) is \(\boxed{3628800}\).

### LlaMA3-Groq

```
python tool_groq.py --temp 0 -m /path/to/llama3-groq-tool-8b.bin
    ________          __  __    __    __  ___
   / ____/ /_  ____ _/ /_/ /   / /   /  |/  /_________  ____
  / /   / __ \/ __ `/ __/ /   / /   / /|_/ // ___/ __ \/ __ \
 / /___/ / / / /_/ / /_/ /___/ /___/ /  / // /__/ /_/ / /_/ /
 \____/_/ /_/\__,_/\__/_____/_____/_/  /_(_)___/ .___/ .___/
You are served by LlaMA-Groq-Tool-Use,        /_/   /_/
with 8030310400 (8.0B) parameters.

You  > weather in beijing?
A.I. >
[Use Tool]: get_weather
The current weather in Beijing is sunny with a temperature of 40°C. It feels like 46°C with a humidity of 35%.
You  > I want to goto New York. Should I bring an umbrella?
A.I. >
[Use Tool]: get_weather
The current weather in New York is partly cloudy with a temperature of 24°C. It feels like 26°C with a humidity of 79%. There's no indication of rain, so you might not need an umbrella.
```