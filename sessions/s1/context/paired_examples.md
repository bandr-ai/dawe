## parallel_0 (parallel)

### v1
**system:**
```
You are a helpful assistant that can call functions.

Available functions (JSON schema):
[
  {
    "name": "spotify.play",
    "description": "Play specific tracks from a given artist for a specific time duration.",
    "parameters": {
      "type": "dict",
      "properties": {
        "artist": {
          "type": "string",
          "description": "The artist whose songs you want to play."
        },
        "duration": {
          "type": "integer",
          "description": "The duration for which the songs should be played, in minutes."
        }
      },
      "required": [
        "artist",
        "duration"
      ]
    }
  }
]

Reply with only a JSON object: {"tool_calls": [{"name": "<function>", "arguments": {...}}]}. Put several calls in the list when needed.
```
**user:** Play songs from the artists Taylor Swift and Maroon 5, with a play time of 20 minutes and 15 minutes respectively, on Spotify.

**gold reply:**
```
{"tool_calls": [{"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}}, {"name": "spotify.play", "arguments": {"artist": "Maroon 5", "duration": 15}}]}
```

### v4_format
**system:**
```
You are a helpful assistant that can call functions.

Available functions (JSON schema):
[
  {
    "name": "spotify.play",
    "description": "Play specific tracks from a given artist for a specific time duration.",
    "parameters": {
      "type": "dict",
      "properties": {
        "artist": {
          "type": "string",
          "description": "The artist whose songs you want to play."
        },
        "duration": {
          "type": "integer",
          "description": "The duration for which the songs should be played, in minutes."
        }
      },
      "required": [
        "artist",
        "duration"
      ]
    }
  }
]

Reply with only tool calls, one per block:
<tool_call name="<function>">
<arg name="<argument>"><JSON value></arg>
</tool_call>
```
**user:** Play songs from the artists Taylor Swift and Maroon 5, with a play time of 20 minutes and 15 minutes respectively, on Spotify.

**gold reply:**
```
<tool_call name="spotify.play">
<arg name="artist">"Taylor Swift"</arg>
<arg name="duration">20</arg>
</tool_call>
<tool_call name="spotify.play">
<arg name="artist">"Maroon 5"</arg>
<arg name="duration">15</arg>
</tool_call>
```

## parallel_1 (parallel)

### v1
**system:**
```
You are a helpful assistant that can call functions.

Available functions (JSON schema):
[
  {
    "name": "calculate_em_force",
    "description": "Calculate the induced electromagnetic force based on Faraday's Law of Electromagnetic Induction, given the magnetic field (in Tesla), change in magnetic field area (in square meters), and the change in time (in seconds).",
    "parameters": {
      "type": "dict",
      "properties": {
        "b_field": {
          "type": "integer",
          "description": "The magnetic field in Tesla."
        },
        "area": {
          "type": "integer",
          "description": "The change in area of magnetic field in square meters."
        },
        "d_time": {
          "type": "integer",
          "description": "The change in time in seconds."
        }
      },
      "required": [
        "b_field",
        "area",
        "d_time"
      ]
    }
  }
]

Reply with only a JSON object: {"tool_calls": [{"name": "<function>", "arguments": {...}}]}. Put several calls in the list when needed.
```
**user:** Calculate the induced electromagnetic force for a magnetic field of 5 Tesla, area of 2 square meters and change in time of 4 seconds, then repeat with a change in time of 10 seconds.

**gold reply:**
```
{"tool_calls": [{"name": "calculate_em_force", "arguments": {"b_field": 5, "area": 2, "d_time": 4}}, {"name": "calculate_em_force", "arguments": {"b_field": 5, "area": 2, "d_time": 10}}]}
```

### v4_format
**system:**
```
You are a helpful assistant that can call functions.

Available functions (JSON schema):
[
  {
    "name": "calculate_em_force",
    "description": "Calculate the induced electromagnetic force based on Faraday's Law of Electromagnetic Induction, given the magnetic field (in Tesla), change in magnetic field area (in square meters), and the change in time (in seconds).",
    "parameters": {
      "type": "dict",
      "properties": {
        "b_field": {
          "type": "integer",
          "description": "The magnetic field in Tesla."
        },
        "area": {
          "type": "integer",
          "description": "The change in area of magnetic field in square meters."
        },
        "d_time": {
          "type": "integer",
          "description": "The change in time in seconds."
        }
      },
      "required": [
        "b_field",
        "area",
        "d_time"
      ]
    }
  }
]

Reply with only tool calls, one per block:
<tool_call name="<function>">
<arg name="<argument>"><JSON value></arg>
</tool_call>
```
**user:** Calculate the induced electromagnetic force for a magnetic field of 5 Tesla, area of 2 square meters and change in time of 4 seconds, then repeat with a change in time of 10 seconds.

**gold reply:**
```
<tool_call name="calculate_em_force">
<arg name="b_field">5</arg>
<arg name="area">2</arg>
<arg name="d_time">4</arg>
</tool_call>
<tool_call name="calculate_em_force">
<arg name="b_field">5</arg>
<arg name="area">2</arg>
<arg name="d_time">10</arg>
</tool_call>
```

## parallel_2 (parallel)

### v1
**system:**
```
You are a helpful assistant that can call functions.

Available functions (JSON schema):
[
  {
    "name": "calculate_resistance",
    "description": "Calculate the resistance of a wire using resistivity, length, and cross-sectional area.",
    "parameters": {
      "type": "dict",
      "properties": {
        "length": {
          "type": "integer",
          "description": "The length of the wire in meters."
        },
        "area": {
          "type": "float",
          "description": "The cross-sectional area of the wire in square meters."
        },
        "resistivity": {
          "type": "string",
          "description": "Resistivity of the material (Default: 'copper'). Allowed values: 'copper', 'aluminum'"
        }
      },
      "required": [
        "length",
        "area"
      ]
    }
  }
]

Reply with only a JSON object: {"tool_calls": [{"name": "<function>", "arguments": {...}}]}. Put several calls in the list when needed.
```
**user:** Calculate the resistance of a wire with a length of 5m and cross sectional area 0.01m² with resistivity of copper and aluminum

**gold reply:**
```
{"tool_calls": [{"name": "calculate_resistance", "arguments": {"length": 5, "area": 0.01, "resistivity": "copper"}}, {"name": "calculate_resistance", "arguments": {"length": 5, "area": 0.01, "resistivity": "aluminum"}}]}
```

### v4_format
**system:**
```
You are a helpful assistant that can call functions.

Available functions (JSON schema):
[
  {
    "name": "calculate_resistance",
    "description": "Calculate the resistance of a wire using resistivity, length, and cross-sectional area.",
    "parameters": {
      "type": "dict",
      "properties": {
        "length": {
          "type": "integer",
          "description": "The length of the wire in meters."
        },
        "area": {
          "type": "float",
          "description": "The cross-sectional area of the wire in square meters."
        },
        "resistivity": {
          "type": "string",
          "description": "Resistivity of the material (Default: 'copper'). Allowed values: 'copper', 'aluminum'"
        }
      },
      "required": [
        "length",
        "area"
      ]
    }
  }
]

Reply with only tool calls, one per block:
<tool_call name="<function>">
<arg name="<argument>"><JSON value></arg>
</tool_call>
```
**user:** Calculate the resistance of a wire with a length of 5m and cross sectional area 0.01m² with resistivity of copper and aluminum

**gold reply:**
```
<tool_call name="calculate_resistance">
<arg name="length">5</arg>
<arg name="area">0.01</arg>
<arg name="resistivity">"copper"</arg>
</tool_call>
<tool_call name="calculate_resistance">
<arg name="length">5</arg>
<arg name="area">0.01</arg>
<arg name="resistivity">"aluminum"</arg>
</tool_call>
```

## simple_0 (simple)

### v1
**system:**
```
You are a helpful assistant that can call functions.

Available functions (JSON schema):
[
  {
    "name": "calculate_triangle_area",
    "description": "Calculate the area of a triangle given its base and height.",
    "parameters": {
      "type": "dict",
      "properties": {
        "base": {
          "type": "integer",
          "description": "The base of the triangle."
        },
        "height": {
          "type": "integer",
          "description": "The height of the triangle."
        },
        "unit": {
          "type": "string",
          "description": "The unit of measure (defaults to 'units' if not specified)"
        }
      },
      "required": [
        "base",
        "height"
      ]
    }
  }
]

Reply with only a JSON object: {"tool_calls": [{"name": "<function>", "arguments": {...}}]}. Put several calls in the list when needed.
```
**user:** Find the area of a triangle with a base of 10 units and height of 5 units.

**gold reply:**
```
{"tool_calls": [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}]}
```

### v4_format
**system:**
```
You are a helpful assistant that can call functions.

Available functions (JSON schema):
[
  {
    "name": "calculate_triangle_area",
    "description": "Calculate the area of a triangle given its base and height.",
    "parameters": {
      "type": "dict",
      "properties": {
        "base": {
          "type": "integer",
          "description": "The base of the triangle."
        },
        "height": {
          "type": "integer",
          "description": "The height of the triangle."
        },
        "unit": {
          "type": "string",
          "description": "The unit of measure (defaults to 'units' if not specified)"
        }
      },
      "required": [
        "base",
        "height"
      ]
    }
  }
]

Reply with only tool calls, one per block:
<tool_call name="<function>">
<arg name="<argument>"><JSON value></arg>
</tool_call>
```
**user:** Find the area of a triangle with a base of 10 units and height of 5 units.

**gold reply:**
```
<tool_call name="calculate_triangle_area">
<arg name="base">10</arg>
<arg name="height">5</arg>
<arg name="unit">"units"</arg>
</tool_call>
```
