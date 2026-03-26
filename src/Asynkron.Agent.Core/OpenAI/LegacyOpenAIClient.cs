using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Asynkron.Agent.Core.Schema;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Asynkron.Agent.Core.Runtime;

/// <summary>
/// LegacyOpenAIClient wraps the HTTP client required to call the legacy OpenAI Chat Completions API.
/// This client uses the /v1/chat/completions endpoint instead of the newer /v1/responses endpoint.
/// </summary>
public sealed class LegacyOpenAIClient : IOpenAIClient
{
    private readonly string _apiKey;
    private readonly string _model;
    private readonly HttpClient _httpClient;
    private readonly PlanSchema.ToolDefinition _tool;
    private readonly string _baseUrl;
    private readonly ILogger _logger;
    private readonly RetryConfig? _retryConfig;

    private const string DefaultOpenAIBaseUrl = "https://api.openai.com/v1";

    public LegacyOpenAIClient(
        string apiKey,
        string model,
        string baseUrl,
        ILogger logger,
        RetryConfig? retryConfig,
        TimeSpan httpTimeout)
    {
        if (string.IsNullOrEmpty(apiKey))
            throw new ArgumentException("openai: API key is required", nameof(apiKey));

        if (string.IsNullOrEmpty(model))
            throw new ArgumentException("openai: model is required", nameof(model));

        baseUrl = (baseUrl ?? "").Trim();
        if (string.IsNullOrEmpty(baseUrl))
            baseUrl = DefaultOpenAIBaseUrl;

        var tool = PlanSchema.GetDefinition();

        _apiKey = apiKey;
        _model = model;
        _httpClient = new HttpClient { Timeout = httpTimeout };
        _tool = tool;
        _baseUrl = baseUrl;
        _logger = logger ?? NullLogger.Instance;
        _retryConfig = retryConfig;
    }

    /// <summary>
    /// RequestPlanAsync sends the accumulated chat history to OpenAI using the legacy Chat Completions API
    /// and returns the resulting tool call payload.
    /// </summary>
    public Task<ToolCall> RequestPlanAsync(CancellationToken ctx, List<ChatMessage> history)
    {
        // Non-streaming path reuses the streaming implementation without emitting deltas.
        return RequestPlanStreamingAsync(ctx, history, null);
    }

    /// <summary>
    /// RequestPlanStreamingAsync streams using the legacy OpenAI Chat Completions API (/v1/chat/completions).
    /// It maps response delta content chunks to the onDelta callback and collects tool_call deltas
    /// into a ToolCall to return on completion.
    /// </summary>
    public async Task<ToolCall> RequestPlanStreamingAsync(
        CancellationToken ctx,
        List<ChatMessage> history,
        Action<string>? onDelta)
    {
        var start = DateTime.UtcNow;
        _logger.LogDebug("Requesting plan from OpenAI (Legacy API). Model={Model} HistoryLength={HistoryLength}", 
            _model, history.Count);

        // Optional debug streaming: set GOAGENT_DEBUG_STREAM=1 to enable verbose prints
        var debugStream = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GOAGENT_DEBUG_STREAM"));
        if (debugStream)
        {
            Console.WriteLine("====== LEGACY STREAM: entering RequestPlanStreamingAsync");
        }

        // Build request
        var messages = BuildMessagesFromHistory(history);
        var payload = BuildRequestBody(messages);

        // Execute request with retry logic
        var resp = await ExecuteRequestAsync(ctx, payload, start, _retryConfig);
        
        try
        {
            // Parse stream
            using var reader = new StreamReader(resp.Content.ReadAsStream());
            var parser = new LegacyOpenAIStreamParser(reader, onDelta, debugStream);
            var toolCall = await parser.ParseAsync();

            var duration = DateTime.UtcNow - start;

            if (!string.IsNullOrEmpty(toolCall.Name))
            {
                _logger.LogDebug("OpenAI (Legacy) API request completed successfully. DurationMs={DurationMs} ToolName={ToolName}",
                    duration.TotalMilliseconds,
                    toolCall.Name);
            }
            else
            {
                _logger.LogDebug("OpenAI (Legacy) API request completed (no tool call). DurationMs={DurationMs}", 
                    duration.TotalMilliseconds);
            }

            return toolCall;
        }
        catch (Exception ex)
        {
            var duration = DateTime.UtcNow - start;
            _logger.LogError(ex, "OpenAI (Legacy) API stream parsing failed. DurationMs={DurationMs} Model={Model}",
                duration.TotalMilliseconds,
                _model);
            throw new Exception($"openai(legacy): stream parsing failed: {ex.Message}", ex);
        }
        finally
        {
            resp.Dispose();
        }
    }

    private static List<Dictionary<string, object>> BuildMessagesFromHistory(List<ChatMessage> history)
    {
        var messages = new List<Dictionary<string, object>>(history.Count);
        foreach (var m in history)
        {
            // Map MessageRole to legacy API roles
            var role = MapRoleToString(m.Role);

            var msg = new Dictionary<string, object>
            {
                ["role"] = role,
                ["content"] = m.Content
            };

            messages.Add(msg);
        }
        return messages;
    }

    private static string MapRoleToString(MessageRole role)
    {
        return role switch
        {
            MessageRole.System => "system",
            MessageRole.User => "user",
            MessageRole.Assistant => "assistant",
            MessageRole.Tool => "user", // Legacy API uses "user" for tool responses
            _ => "user"
        };
    }

    private byte[] BuildRequestBody(List<Dictionary<string, object>> messages)
    {
        // Legacy Chat Completions API format
        var reqBody = new Dictionary<string, object>
        {
            ["model"] = _model,
            ["messages"] = messages,
            ["stream"] = true,
            // Define the function tool in the legacy API format
            ["tools"] = new List<Dictionary<string, object>>
            {
                new()
                {
                    ["type"] = "function",
                    ["function"] = new Dictionary<string, object>
                    {
                        ["name"] = _tool.Name,
                        ["description"] = _tool.Description,
                        ["parameters"] = _tool.Parameters
                    }
                }
            },
            // Require a tool call; with only one tool defined, this forces the model
            // to call our tool with arguments.
            ["tool_choice"] = new Dictionary<string, object>
            {
                ["type"] = "function",
                ["function"] = new Dictionary<string, object>
                {
                    ["name"] = _tool.Name
                }
            }
        };

        return System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(reqBody);
    }

    private async Task<HttpResponseMessage> ExecuteRequestAsync(
        CancellationToken ctx,
        byte[] payload,
        DateTime start,
        RetryConfig? retryConfig)
    {
        HttpResponseMessage? resp = null;
        Exception? lastErr = null;

        await RetryHelper.ExecuteWithRetry(_retryConfig, async () =>
        {
            // Create new request for each retry attempt
            var apiRoot = _baseUrl.TrimEnd('/');
            var url = $"{apiRoot}/chat/completions";

            var req = new HttpRequestMessage(HttpMethod.Post, url)
            {
                Content = new ByteArrayContent(payload)
            };
            req.Headers.Add("Authorization", $"Bearer {_apiKey}");
            req.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");

            try
            {
                resp = await _httpClient.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ctx);

                if (!resp.IsSuccessStatusCode)
                {
                    var msg = await resp.Content.ReadAsStringAsync(ctx);
                    if (msg.Length > 4096)
                        msg = msg[..4096];
                    
                    resp.Dispose();
                    var duration = DateTime.UtcNow - start;
                    var retryable = RetryHelper.IsRetryableStatusCode((int)resp.StatusCode);

                    var statusError = new RetryableApiError(
                        $"openai(legacy): status {resp.StatusCode}: {msg}",
                        (int)resp.StatusCode,
                        retryable);
                    _logger.LogError(statusError, "OpenAI (Legacy) API returned error status. StatusCode={StatusCode} DurationMs={DurationMs} Retryable={Retryable}",
                        (int)resp.StatusCode,
                        duration.TotalMilliseconds,
                        retryable);
                    lastErr = statusError;
                    resp = null;
                    throw lastErr;
                }
            }
            catch (HttpRequestException ex)
            {
                var duration = DateTime.UtcNow - start;
                var retryable = RetryHelper.IsRetryableError(ex);
                _logger.LogError(ex, "OpenAI (Legacy) API request failed. Url={Url} DurationMs={DurationMs} Retryable={Retryable}",
                    url,
                    duration.TotalMilliseconds,
                    retryable);

                lastErr = new RetryableApiError(
                    $"openai(legacy): do request: {ex.Message}",
                    0,
                    retryable,
                    ex);
                throw lastErr;
            }
        }, ctx);

        if (resp == null)
        {
            var duration = DateTime.UtcNow - start;
            if (lastErr != null)
                throw lastErr;
            throw new Exception("openai(legacy): request failed");
        }

        return resp;
    }
}