using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace Asynkron.Agent.Core.Runtime;

/// <summary>
/// LegacyOpenAIStreamParser handles parsing of SSE (Server-Sent Events) streams from
/// the legacy OpenAI Chat Completions API (/v1/chat/completions).
/// </summary>
internal sealed class LegacyOpenAIStreamParser(StreamReader reader, Action<string>? onDelta, bool debugStream)
{
    private string _toolId = "";
    private string _toolName = "";
    private string _toolArgs = "";
    private int _lastEmittedMessage = 0;

    /// <summary>
    /// ParseAsync reads and parses the SSE stream until completion or error.
    /// </summary>
    public async Task<ToolCall> ParseAsync()
    {
        if (debugStream)
        {
            Console.WriteLine("====== LEGACY STREAM: HTTP connected; starting SSE read loop");
        }

        while (true)
        {
            var line = await reader.ReadLineAsync();
            if (line == null)
            {
                break; // EOF
            }

            line = line.Trim();
            if (string.IsNullOrEmpty(line) || line.StartsWith(":"))
            {
                continue; // keepalive/comment
            }

            if (!line.StartsWith("data:") && !line.StartsWith("data: "))
            {
                continue;
            }

            var chunkData = line.StartsWith("data: ")
                ? line["data: ".Length..].Trim()
                : line["data:".Length..].Trim();

            if (chunkData == "[DONE]")
            {
                if (debugStream)
                {
                    Console.WriteLine("------ LEGACY STREAM: [DONE]");
                }
                break;
            }

            var evt = ParseEvent(chunkData);
            if (evt == null)
            {
                continue;
            }

            ProcessEvent(evt);
        }

        if (!string.IsNullOrEmpty(_toolName))
        {
            return new ToolCall { ID = _toolId, Name = _toolName, Arguments = _toolArgs };
        }

        // No tool call is valid for plain text responses
        return new ToolCall();
    }

    /// <summary>
    /// ParseEvent parses a single SSE data chunk into an event map.
    /// </summary>
    private Dictionary<string, object>? ParseEvent(string chunkData)
    {
        try
        {
            var evt = JsonSerializer.Deserialize<Dictionary<string, object>>(chunkData);
            if (debugStream)
            {
                Console.WriteLine("------ LEGACY STREAM: event received");
            }
            return evt;
        }
        catch (JsonException ex)
        {
            if (debugStream)
            {
                var chunkPreview = chunkData.Length > 200 ? chunkData[..200] + "..." : chunkData;
                Console.WriteLine($"------ LEGACY STREAM: decode-error {ex.Message} (chunk: \"{chunkPreview}\")");
            }
            return null;
        }
    }

    /// <summary>
    /// ProcessEvent handles a single stream event and updates parser state.
    /// </summary>
    private void ProcessEvent(Dictionary<string, object> evt)
    {
        // Legacy Chat Completions API format:
        // {
        //   "id": "...",
        //   "object": "chat.completion.chunk",
        //   "created": 1234567890,
        //   "model": "gpt-4",
        //   "choices": [
        //     {
        //       "index": 0,
        //       "delta": {
        //         "content": "text" OR "tool_calls": [...]
        //       },
        //       "finish_reason": null
        //     }
        //   ]
        // }

        if (!evt.TryGetValue("choices", out var choicesObj))
            return;

        if (!TryGetArray(choicesObj, out var choices))
            return;

        if (choices.Count == 0)
            return;

        var choice = choices[0];
        if (!TryGetDictionary(choice, out var choiceDict))
            return;

        if (!choiceDict.TryGetValue("delta", out var deltaObj))
            return;

        if (!TryGetDictionary(deltaObj, out var deltaDict))
            return;

        // Process content delta (text output)
        if (deltaDict.TryGetValue("content", out var contentObj))
        {
            var content = contentObj?.ToString() ?? "";
            if (!string.IsNullOrEmpty(content))
            {
                onDelta?.Invoke(content);
            }
        }

        // Process tool_calls delta
        if (deltaDict.TryGetValue("tool_calls", out var toolCallsObj))
        {
            if (TryGetArray(toolCallsObj, out var toolCalls))
            {
                foreach (var toolCallObj in toolCalls)
                {
                    if (TryGetDictionary(toolCallObj, out var toolCall))
                    {
                        ProcessToolCallDelta(toolCall);
                    }
                }
            }
        }
    }

    /// <summary>
    /// ProcessToolCallDelta processes a single tool call delta from the legacy API.
    /// </summary>
    private void ProcessToolCallDelta(Dictionary<string, object> toolCall)
    {
        // Legacy API tool call format:
        // {
        //   "index": 0,
        //   "id": "call_xxx",
        //   "type": "function",
        //   "function": {
        //     "name": "function_name",
        //     "arguments": "{..."
        //   }
        // }

        if (toolCall.TryGetValue("id", out var idObj))
        {
            var id = idObj?.ToString() ?? "";
            if (!string.IsNullOrEmpty(id) && id != _toolId)
            {
                _toolId = id;
                _toolArgs = "";
                _lastEmittedMessage = 0;
            }
        }

        if (toolCall.TryGetValue("function", out var functionObj))
        {
            if (TryGetDictionary(functionObj, out var functionDict))
            {
                if (functionDict.TryGetValue("name", out var nameObj))
                {
                    var name = nameObj?.ToString() ?? "";
                    if (!string.IsNullOrEmpty(name))
                    {
                        _toolName = name;
                    }
                }

                if (functionDict.TryGetValue("arguments", out var argsObj))
                {
                    var args = argsObj?.ToString() ?? "";
                    if (!string.IsNullOrEmpty(args))
                    {
                        _toolArgs += args;
                        EmitMessageDelta(_toolArgs);
                    }
                }
            }
        }
    }

    /// <summary>
    /// EmitMessageDelta extracts and emits the "message" field from partial JSON.
    /// </summary>
    private void EmitMessageDelta(string buf)
    {
        if (onDelta == null)
            return;

        var (raw, _, ok) = OpenAIClient.ExtractPartialJSONStringField(buf, "message");
        if (!ok)
            return;

        var decoded = OpenAIClient.DecodePartialJSONString(raw);
        if (string.IsNullOrEmpty(decoded))
            return;

        // Emit only the new part since last emission
        var currentLength = decoded.Length;
        if (currentLength > _lastEmittedMessage)
        {
            onDelta(decoded[_lastEmittedMessage..]);
            _lastEmittedMessage = currentLength;
        }
    }

    private static bool TryGetDictionary(object? obj, out Dictionary<string, object> dict)
    {
        dict = new Dictionary<string, object>();
        
        if (obj is Dictionary<string, object> d)
        {
            dict = d;
            return true;
        }

        if (obj is JsonElement { ValueKind: JsonValueKind.Object } elem)
        {
            dict = new Dictionary<string, object>();
            foreach (var prop in elem.EnumerateObject())
            {
                dict[prop.Name] = prop.Value;
            }
            return true;
        }

        return false;
    }

    private static bool TryGetArray(object? obj, out List<object> arr)
    {
        arr = [];

        if (obj is List<object> list)
        {
            arr = list;
            return true;
        }

        if (obj is object[] objArr)
        {
            arr = objArr.ToList();
            return true;
        }

        if (obj is JsonElement { ValueKind: JsonValueKind.Array } elem)
        {
            arr = [];
            foreach (var item in elem.EnumerateArray())
            {
                arr.Add(item);
            }
            return true;
        }

        return false;
    }
}