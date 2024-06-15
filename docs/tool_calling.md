# Tool Calling

Some models support tool calling (function calling, or code interpreter). It would be much easier (and cool) to do tool calling in Python.

Some models that can be used for tool calling and demos are provided:

* [ChatGLM3](../bindings/tool_glm3.py)
* [GLM-4](../bindings/tool_glm4.py)

## Precondition

Build target [`libchatllm`](binding.md).

## Session Files

Since tools are generally encoded in system prompt which results a very long prompt, causing a large delay before generating the first output.
Saving the session and reload it later can eliminate such delay.  To save the session exactly after system prompt is evaluated:

```
python ....py -i -m /path/to/your/model.bin --save_session 0 /path/to/session/file
```

## Demo

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