from vllm import LLM

from pl_mai_vps.util.print_util import print_segment

# 50 needs about 14 GB of VRAM
frame_amount = 50
# Approximately N images of size + 100 tokens output
# 480p = 854x480 pixels
# 28x28 pixels ~= 1 token => 480p ~= 31 x 18 tokens ~= 558 tokens
max_model_len = 558 * frame_amount + 100

print(
    f"Trying to load Qwen2.5-VL-3B-Instruct model with space for {frame_amount}x 480p frames and 100 output tokens...")

# Initialize the vLLM engine.
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    max_model_len=max_model_len,
    # No prefix caching, will also not be used in experiments
    # Only make sense for chat conversations or static prompts
    enable_prefix_caching=False,
    limit_mm_per_prompt={"image": frame_amount}
)

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant",
    },
    {
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/7/70/Oftheunicorn.jpg"
            }
        }, {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Domenichounicorndetail.jpg/1024px-Domenichounicorndetail.jpg"
            }
        }]
    }
]

print_segment([
    "Type message and press enter to chat with model ('exit' for quitting the app)",
    "There are currently two images of unicorns in the context"
])

while True:
    line = input()

    if line.lower() == "exit":
        break

    conversation.append(
        {
            "role": "user",
            "content": line,
        }
    )

    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 500
    outputs = llm.chat(conversation, sampling_params)

    assert len(outputs) == 1
    output = outputs[0]

    generated_text = output.outputs[0].text
    print("=" * 5)
    print(generated_text)

    conversation.append({
        "role": "assistant",
        "content": generated_text,
    })
