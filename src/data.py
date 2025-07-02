from transformers import GenerationConfig


generation_config = GenerationConfig(
    # do_sample=False,
    temperature=0.1,
    top_p=0.95,
    seed=42,
)

def make_prompt(front, back, tokenizer, flag_vlm=False):
    instruction = (
        "[지침]\n제시된 '앞 문장'과 '뒤 문장'의 내용을 바탕으로 두 문장 사이의 관계에 대한 label을 생성하시오. [생성 기준]을 꼼꼼히 읽고 이해하는 것이 중요합니다.\n\n"
        "[생성 기준]\n"
        "1 - 당신은 제시된 '앞 문장'과 '뒤 문장'의 내용을 바탕으로 두 문장 사이의 관계를 분류하는 챗봇입니다.\n"
        "2 - 분류 결과인 label의 종류는 '순접', '역접', '양립'입니다.\n"
        "3 - '순접'은 앞 문장의 내용이 원인·조건·배경이 되어, 뒤 문장이 자연스럽게 이어지는 관계입니다.\n"
        "4 - '역접'은 앞 문장의 내용과 뒤 문장이 대조되거나, 예상과는 다른 방향으로 전개되는 관계입니다.\n"
        "5 - '양립'은 순접과 역접이 모두 가능한 문장을 나타냅니다.\n"
        "6 - 출력은 '순접', '역접', '양립' 중에서 1개만 생성하시오."
    )
    sentence = f"[문장]\n - 앞 문장: {front}\n - 뒤 문장: {back}"
    user_prompt = instruction + "\n\n" + sentence

    # LLM
    if not flag_vlm:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
    # Vision-Language 
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, # Qwen3, Switches between thinking and non-thinking modes. Default is True.
        return_tensors="pt",
        # return_dict=True, # tokenize=True인 경우에 사용
    )

    return prompt
