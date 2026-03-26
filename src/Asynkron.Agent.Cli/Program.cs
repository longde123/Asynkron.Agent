using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Asynkron.Agent.Core.Runtime;

namespace Asynkron.Agent.Cli;

public class Program
{
    public static async Task<int> Main(string[] args)
    {
        return await RunAsync(args, Console.Out, Console.Error, CancellationToken.None);
    }

    public static async Task<int> RunAsync(string[] args, TextWriter stdout, TextWriter stderr, CancellationToken cancellationToken)
    {
        var envPath = Path.Combine(Directory.GetCurrentDirectory(), ".env");
        if (File.Exists(envPath))
        {
            try
            {
                foreach (var line in File.ReadAllLines(envPath))
                {
                    var trimmed = line.Trim();
                    if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#'))
                        continue;

                    var parts = trimmed.Split('=', 2);
                    if (parts.Length == 2)
                    {
                        Environment.SetEnvironmentVariable(parts[0].Trim(), parts[1].Trim());
                    }
                }
            }
            catch (Exception ex)
            {
                await stderr.WriteLineAsync($"failed to load .env: {ex.Message}");
                return 1;
            }
        }

        var defaultModel = Environment.GetEnvironmentVariable("OPENAI_MODEL") ?? "gpt-4o";
        var defaultReasoning = Environment.GetEnvironmentVariable("OPENAI_REASONING_EFFORT") ?? "";
        var defaultBaseURL = Environment.GetEnvironmentVariable("OPENAI_BASE_URL") ?? "";
        var defaultApiVersion = Environment.GetEnvironmentVariable("OPENAI_API_VERSION") ?? "responses";
        var defaultHeadless = Environment.GetEnvironmentVariable("HEADLESS");
        var defaultDisableInputReader = Environment.GetEnvironmentVariable("DISABLE_INPUT_READER");
        
        string? model = null, reasoningEffort = null, promptAugmentation = null, baseURL = null, prompt = null, research = null, apiVersion = null;
        bool? headlessFlag = null, interactiveFlag = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model" when i + 1 < args.Length: model = args[++i]; break;
                case "--reasoning-effort" when i + 1 < args.Length: reasoningEffort = args[++i]; break;
                case "--augment" when i + 1 < args.Length: promptAugmentation = args[++i]; break;
                case "--openai-base-url" when i + 1 < args.Length: baseURL = args[++i]; break;
                case "--api-version" when i + 1 < args.Length: apiVersion = args[++i]; break;
                case "--prompt" when i + 1 < args.Length: prompt = args[++i]; break;
                case "--research" when i + 1 < args.Length: research = args[++i]; break;
                case "--headless":
                    headlessFlag = true;
                    break;
                case "--interactive":
                    interactiveFlag = true;
                    break;
                case "--help":
                case "-h":
                    await stdout.WriteLineAsync("Usage: goagent [options]");
                    await stdout.WriteLineAsync("Options:");
                    await stdout.WriteLineAsync("  --model MODEL              Model to use (default: from env or gpt-4o)");
                    await stdout.WriteLineAsync("  --api-version VERSION     API version: responses or chat_completions (default: from env or responses)");
                    await stdout.WriteLineAsync("  --reasoning-effort LEVEL  Reasoning effort: low, medium, high (for o1 models)");
                    await stdout.WriteLineAsync("  --augment TEXT            System prompt augmentation");
                    await stdout.WriteLineAsync("  --openai-base-url URL     Custom OpenAI API base URL");
                    await stdout.WriteLineAsync("  --prompt TEXT             Run a single prompt in hands-free mode");
                    await stdout.WriteLineAsync("  --research JSON           Run research (JSON with 'goal' and 'turns')");
                    await stdout.WriteLineAsync("  --headless                Run in headless mode (disable interactive input)");
                    await stdout.WriteLineAsync("  --interactive             Force interactive mode (enable input even with --prompt)");
                    await stdout.WriteLineAsync("  --help, -h                Show this help message");
                    return 0;
                default:
                    await stderr.WriteLineAsync($"Unknown argument: {args[i]}");
                    return 2;
            }
        }

        model ??= defaultModel;
        reasoningEffort ??= defaultReasoning;
        baseURL ??= defaultBaseURL;
        apiVersion ??= defaultApiVersion;

        // Parse ApiVersion from string
        var parsedApiVersion = apiVersion.ToLowerInvariant() switch
        {
            "chat_completions" or "legacy" or "old" => ApiVersion.ChatCompletions,
            "responses" or "new" or "" => ApiVersion.Responses,
            _ => ApiVersion.Responses  // Default to Responses API
        };

        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrEmpty(apiKey))
        {
            await stderr.WriteLineAsync("OPENAI_API_KEY must be set in the environment.");
            return 1;
        }

        var combinedAugment = (promptAugmentation ?? "").Trim();

        // Determine if input reader should be disabled based on priority
        bool disableInputReader = false;
        
        // Priority 1: Command line flags
        if (headlessFlag == true)
        {
            disableInputReader = true;
        }
        else if (interactiveFlag == true)
        {
            disableInputReader = false;
        }
        // Priority 2: Environment variables
        else if (!string.IsNullOrEmpty(defaultHeadless) && bool.TryParse(defaultHeadless, out var headlessEnv))
        {
            disableInputReader = headlessEnv;
        }
        else if (!string.IsNullOrEmpty(defaultDisableInputReader) && bool.TryParse(defaultDisableInputReader, out var disableInputEnv))
        {
            disableInputReader = disableInputEnv;
        }
        // Priority 3: Default behavior - disable for prompt/research mode, enable for interactive
        else if (!string.IsNullOrWhiteSpace(prompt) || !string.IsNullOrWhiteSpace(research))
        {
            disableInputReader = true;
        }
        else
        {
            disableInputReader = false; // Default to interactive mode
        }

        var options = new RuntimeOptions
        {
            ApiKey = apiKey,
            ApiBaseUrl = baseURL.Trim(),
            Model = model,
            ApiVersion = parsedApiVersion,
            ReasoningEffort = reasoningEffort,
            SystemPromptAugment = combinedAugment,
            DisableOutputForwarding = true,
            DisableInputReader = disableInputReader,
            UseStreaming = true
        };

        if (!string.IsNullOrWhiteSpace(research))
        {
            try
            {
                var spec = JsonSerializer.Deserialize<ResearchSpec>(research);
                if (spec == null || string.IsNullOrWhiteSpace(spec.Goal))
                {
                    await stderr.WriteLineAsync("--research requires non-empty goal");
                    return 2;
                }

                // Use 'with' expression for immutable record
                options = options with
                {
                    HandsFree = true,
                    HandsFreeTopic = spec.Goal.Trim(),
                    MaxPasses = spec.Turns > 0 ? spec.Turns : options.MaxPasses,
                    HandsFreeAutoReply = $"Please continue to work on the set goal. No human available. Goal: {spec.Goal}"
                };

                return await RunHeadlessResearchAsync(cancellationToken, options, stdout, stderr);
            }
            catch (JsonException ex)
            {
                await stderr.WriteLineAsync($"invalid --research JSON: {ex.Message}");
                return 2;
            }
        }
        else if (!string.IsNullOrWhiteSpace(prompt))
        {
            options = options with
            {
                HandsFree = true,
                HandsFreeTopic = prompt.Trim()
            };
        }

        return await RunHeadlessAsync(cancellationToken, options, stdout, stderr);
    }

    private static async Task<int> RunHeadlessAsync(CancellationToken ctx, RuntimeOptions options, TextWriter stdout, TextWriter stderr)
    {
        options = options with
        {
            UseStreaming = true,
            DisableOutputForwarding = true
            // DisableInputReader is already set by the caller based on configuration
        };

        Asynkron.Agent.Core.Runtime.Runtime agent;
        try
        {
            agent = Asynkron.Agent.Core.Runtime.Runtime.Create(options);
        }
        catch (Exception ex)
        {
            await stderr.WriteLineAsync($"failed to create runtime: {ex.Message}");
            return 1;
        }

        var outputs = agent.Outputs();
        var runTask = agent.RunAsync(ctx);

        await foreach (var evt in outputs.ReadAllAsync(ctx))
        {
            await stdout.WriteLineAsync($"[{evt.Type}] {evt.Message}");
        }

        var error = await runTask;
        return error == null ? 0 : 1;
    }

    private static async Task<int> RunHeadlessResearchAsync(CancellationToken ctx, RuntimeOptions options, TextWriter stdout, TextWriter stderr)
    {
        options = options with
        {
            UseStreaming = true,
            DisableOutputForwarding = true,
            DisableInputReader = true
        };

        Asynkron.Agent.Core.Runtime.Runtime agent;
        try
        {
            agent = Asynkron.Agent.Core.Runtime.Runtime.Create(options);
        }
        catch (Exception ex)
        {
            await stderr.WriteLineAsync($"failed to create runtime: {ex.Message}");
            return 1;
        }

        var outputs = agent.Outputs();
        var runTask = agent.RunAsync(ctx);

        string lastAssistant = "";
        bool success = false, failedBudget = false;

        await foreach (var evt in outputs.ReadAllAsync(ctx))
        {
            switch (evt.Type)
            {
                case EventType.AssistantMessage:
                    var msg = evt.Message.Trim();
                    if (!string.IsNullOrEmpty(msg)) lastAssistant = msg;
                    break;
                case EventType.Status:
                    if (evt.Message.Contains("Hands-free session complete")) success = true;
                    break;
                case EventType.Error:
                    if (evt.Message.Contains("Maximum pass limit")) failedBudget = true;
                    break;
            }
        }

        await runTask;

        if (success)
        {
            if (!string.IsNullOrEmpty(lastAssistant)) await stdout.WriteLineAsync(lastAssistant);
            return 0;
        }

        if (!string.IsNullOrEmpty(lastAssistant))
            await stderr.WriteLineAsync(lastAssistant);
        else if (failedBudget)
            await stderr.WriteLineAsync("No solution found within turn budget.");
        else
            await stderr.WriteLineAsync("Agent terminated without a final result.");
        return 1;
    }

    private sealed class ResearchSpec
    {
        public string Goal { get; set; } = "";
        public int Turns { get; set; }
    }
}
