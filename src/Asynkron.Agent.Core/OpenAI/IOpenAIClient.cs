using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Asynkron.Agent.Core.Runtime;

/// <summary>
/// IOpenAIClient defines the common interface for OpenAI API clients.
/// This abstraction allows switching between the modern Responses API and
/// the legacy Chat Completions API without changing the runtime code.
/// </summary>
public interface IOpenAIClient
{
    /// <summary>
    /// RequestPlanAsync sends the accumulated chat history to OpenAI and returns
    /// the resulting tool call payload so the runtime can perform validation before decoding it.
    /// </summary>
    Task<ToolCall> RequestPlanAsync(CancellationToken ctx, List<ChatMessage> history);

    /// <summary>
    /// RequestPlanStreamingAsync streams the OpenAI response. It maps response delta
    /// content chunks to the onDelta callback and collects tool call deltas into a
    /// ToolCall to return on completion.
    /// </summary>
    Task<ToolCall> RequestPlanStreamingAsync(
        CancellationToken ctx,
        List<ChatMessage> history,
        Action<string>? onDelta);
}