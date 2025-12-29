"""
BlockRun connector for Polymarket agents.

Provides LLM access via x402 USDC micropayments on Base network.
Agents can pay for AI inference with their trading wallet - no API keys needed.

SECURITY NOTE - Private Key Handling:
=====================================
Your private key NEVER leaves your machine. Here's what happens:

1. Key stays local - only used to sign an EIP-712 typed data message
2. Only the SIGNATURE is sent in the PAYMENT-SIGNATURE header
3. BlockRun verifies the signature on-chain via Coinbase CDP facilitator
4. Your actual private key is NEVER transmitted to any server

This is the same security model as:
- Signing a MetaMask transaction
- Any on-chain swap or trade
- Polymarket's existing trading flow

The x402 protocol uses EIP-3009 (TransferWithAuthorization) which allows
gasless USDC transfers via signed messages - your key signs locally,
the signature authorizes the transfer on-chain.

Usage:
    from agents.connectors.blockrun import create_blockrun_llm

    llm = create_blockrun_llm(
        model="gpt-4o",
        private_key=os.getenv("POLYGON_WALLET_PRIVATE_KEY")  # Works on Base too
    )
    response = llm.invoke("Hello!")
"""

import os
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Model mapping: common names -> BlockRun model IDs
BLOCKRUN_MODEL_MAP: Dict[str, str] = {
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-4": "openai/gpt-4",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "o1": "openai/o1",
    "o1-mini": "openai/o1-mini",
    "o1-preview": "openai/o1-preview",
    "o3-mini": "openai/o3-mini",
    # Anthropic
    "claude-3-5-sonnet": "anthropic/claude-sonnet-4",
    "claude-3-5-haiku": "anthropic/claude-haiku-4.5",
    "claude-3-opus": "anthropic/claude-opus-4",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-opus-4": "anthropic/claude-opus-4",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    # Google
    "gemini-2.0-flash": "google/gemini-2.0-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-1.5-pro": "google/gemini-1.5-pro",
    "gemini-1.5-flash": "google/gemini-1.5-flash",
    # DeepSeek
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-reasoner": "deepseek/deepseek-reasoner",
}

# Max tokens per model
BLOCKRUN_MAX_TOKENS: Dict[str, int] = {
    "openai/gpt-4o": 128000,
    "openai/gpt-4o-mini": 128000,
    "openai/gpt-4-turbo": 128000,
    "openai/gpt-4": 8192,
    "openai/gpt-3.5-turbo": 16385,
    "openai/o1": 200000,
    "openai/o1-mini": 128000,
    "openai/o3-mini": 200000,
    "anthropic/claude-sonnet-4": 200000,
    "anthropic/claude-opus-4": 200000,
    "anthropic/claude-haiku-4.5": 200000,
    "google/gemini-2.0-flash": 1000000,
    "google/gemini-2.5-pro": 1000000,
    "google/gemini-2.5-flash": 1000000,
    "deepseek/deepseek-chat": 64000,
    "deepseek/deepseek-reasoner": 64000,
}


def get_blockrun_model_name(model: str) -> str:
    """Convert common model names to BlockRun format."""
    if "/" in model:
        return model  # Already in BlockRun format
    return BLOCKRUN_MODEL_MAP.get(model, f"openai/{model}")


class BlockRunChat(BaseChatModel):
    """
    LangChain ChatModel that uses BlockRun with x402 micropayments.

    This properly implements the x402 payment flow:
    1. Send request to BlockRun API
    2. Receive 402 Payment Required with payment details
    3. Sign USDC payment on Base with wallet (LOCAL signing - key never sent)
    4. Retry with PAYMENT-SIGNATURE header (only signature sent, not key)

    Security: Your private key is used ONLY for local EIP-712 signing.
    The key never leaves your machine - only signatures are transmitted.
    """

    model: str = "openai/gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    private_key: Optional[str] = None
    base_url: str = "https://blockrun.ai/api"
    _client: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get private key from param or environment
        # SECURITY: Key is stored in memory only, used for LOCAL signing
        # The key is NEVER transmitted - only EIP-712 signatures are sent
        key = self.private_key or os.getenv("BLOCKRUN_WALLET_KEY") or os.getenv("POLYGON_WALLET_PRIVATE_KEY")
        if not key:
            raise ValueError(
                "Wallet private key required for BlockRun x402 payments. "
                "Set BLOCKRUN_WALLET_KEY or POLYGON_WALLET_PRIVATE_KEY environment variable. "
                "NOTE: Your key never leaves your machine - only signatures are sent."
            )

        # Import and initialize BlockRun client
        # The client holds the key in memory for signing, never transmits it
        try:
            from blockrun_llm import LLMClient
            self._client = LLMClient(private_key=key, api_url=self.base_url)
        except ImportError:
            raise ImportError(
                "blockrun-llm package required for x402 payments. "
                "Install with: pip install blockrun-llm"
            )

    @property
    def _llm_type(self) -> str:
        return "blockrun"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "base_url": self.base_url,
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using BlockRun with x402 payment."""
        # Convert messages
        openai_messages = self._convert_messages(messages)

        # Call BlockRun API (handles 402 payment flow internally)
        response = self._client.chat_completion(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Extract response content
        content = response.choices[0].message.content

        # Build result
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)

        return ChatResult(
            generations=[generation],
            llm_output={
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            }
        )

    def get_wallet_address(self) -> str:
        """Get the wallet address being used for payments."""
        return self._client.get_wallet_address()


def create_blockrun_llm(
    model: str = "gpt-4o",
    temperature: float = 0.7,
    private_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> BlockRunChat:
    """
    Create a BlockRun LLM with x402 micropayments.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash")
        temperature: Sampling temperature (0-1)
        private_key: EVM wallet private key for payments (or set BLOCKRUN_WALLET_KEY env var)
        base_url: BlockRun API URL (default: https://blockrun.ai/api)

    Returns:
        LangChain-compatible ChatModel that pays for inference with USDC on Base

    Example:
        llm = create_blockrun_llm("gpt-4o")
        response = llm.invoke("What is the weather?")
        print(response.content)

    Note:
        Your wallet needs USDC on Base network. Get USDC at https://bridge.base.org
    """
    blockrun_model = get_blockrun_model_name(model)

    return BlockRunChat(
        model=blockrun_model,
        temperature=temperature,
        private_key=private_key,
        base_url=base_url or "https://blockrun.ai/api",
        **kwargs,
    )
