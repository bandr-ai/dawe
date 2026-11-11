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

Before the calls, write exactly one line <plan>...</plan> saying which functions you will call and why. Replies without a plan are rejected.
```
**user:** Play songs from the artists Taylor Swift and Maroon 5, with a play time of 20 minutes and 15 minutes respectively, on Spotify.

**gold reply:**
```
<plan>Call spotify.play, spotify.play.</plan>
{"tool_calls": [{"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}}, {"name": "spotify.play", "arguments": {"artist": "Maroon 5", "duration": 15}}]}
```
