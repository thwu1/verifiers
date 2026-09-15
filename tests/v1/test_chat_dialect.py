from openai.types.chat import ChatCompletion
from verifiers.v1.dialects.chat import response_from_wire


def test_response_from_wire_keeps_gateway_token_extensions() -> None:
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "created": 1,
            "model": "Kimi-K3",
            "object": "chat.completion",
            "prompt_token_ids": [10, 11, 12],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "thinking",
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "think",
                                "bytes": [116],
                                "logprob": -0.25,
                                "top_logprobs": [],
                            },
                            {
                                "token": "ing",
                                "bytes": [105],
                                "logprob": -0.5,
                                "top_logprobs": [],
                            },
                        ]
                    },
                    "provider_specific_fields": {"token_ids": [20, 21]},
                }
            ],
        }
    )

    response = response_from_wire(completion)

    assert response.message.reasoning_content == "thinking"
    assert response.tokens is not None
    assert response.tokens.prompt_ids == [10, 11, 12]
    assert response.tokens.completion_ids == [20, 21]
    assert response.tokens.completion_logprobs == [-0.25, -0.5]
