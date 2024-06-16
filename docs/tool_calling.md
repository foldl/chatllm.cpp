# Tool Calling

Some models support tool calling (function calling, or code interpreter). It would be much easier (and cool) to do tool calling in Python.
Demos of tool calling for these models are provided:

* [ChatGLM3](../bindings/tool_glm3.py)
* [GLM-4](../bindings/tool_glm4.py)
* [Mistral-Instruct-7B-v0.3](../bindings/tool_mistral.py)

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
python tool_glm4.py -i --temp 0 -m path/to/glm-4.bin --load_session /path/to/session/file
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
python tool_mistral.py -i -m ..\quantized\mistral-7b-v0.3.bin
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