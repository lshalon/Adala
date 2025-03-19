from ._litellm import AsyncLiteLLMChatRuntime, AsyncLiteLLMVisionRuntime
from ._openrouter import OpenRouterChatRuntime

# litellm already reads the OPENAI_API_KEY env var, which was the reason for this class
OpenAIChatRuntime = OpenRouterChatRuntime
AsyncOpenAIChatRuntime = AsyncLiteLLMChatRuntime
AsyncOpenAIVisionRuntime = AsyncLiteLLMVisionRuntime
