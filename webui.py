import os
import yaml
import gradio as gr
from typing import List, Tuple, Any, Dict, Optional

from mapping_infer_r1 import DEFAULT_MAPPING_CKPT, MappingChatSession

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

# ========== 配置 =========
DEFAULT_CONFIG = {
    'mapping_ckpt': DEFAULT_MAPPING_CKPT,
    # 提示词: 用户提供 (保持原格式, 不改动其中缺失的引号以便原样实验)
    'prompt_prefix_text': '{"system":你是一个人工智能助手，你现在有阅读图片的能力。所有的图片内容都会以<img name=[name of image]>[image content]</img>的形式呈现。对话中时刻使用中文， 并且无论在任何情况(哪怕是思考中)都不要复述图片内容},\n{"user":现在，请问image1的大概内容是什么?主要颜色是什么?\n<image name=image1>',
    'prompt_suffix_text': '</image>}\n<think>\n</think>'
}

def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                user_cfg = yaml.safe_load(f) or {}
            for k in DEFAULT_CONFIG.keys():
                if k in user_cfg:
                    cfg[k] = user_cfg[k]
        except Exception as e:
            print(f'[Config] 读取失败: {e}')
    else:
        print('[Config] 未找到 config.yaml, 使用默认配置')
    return cfg

# ========== 辅助 ==========

def session_history_to_pairs(history: List[Dict[str,str]]) -> List[Tuple[str,str]]:
    pairs: List[Tuple[str,str]] = []
    pending_user: Optional[str] = None
    for turn in history:
        if turn['role'] == 'user':
            pending_user = turn['content']
        elif turn['role'] == 'assistant' and pending_user is not None:
            pairs.append((pending_user, turn['content']))
            pending_user = None
    return pairs

# ========== 回调 ==========

def reload_config_cb():
    return load_config(), '配置已重新加载'

def ensure_session(session: Optional[MappingChatSession], cfg: Dict[str,Any], prefix_mode, prefix_len, prefix_source, top_k_embed) -> MappingChatSession:
    if session is None or session.mapping_ckpt != cfg['mapping_ckpt']:
        session = MappingChatSession(mapping_ckpt=cfg['mapping_ckpt'], prefix_mode=prefix_mode, prefix_len=prefix_len, prefix_source=prefix_source, top_k_embed=top_k_embed)
    else:
        session.prefix_mode = prefix_mode
        session.prefix_len = prefix_len
        session.prefix_source = prefix_source
        session.top_k_embed = top_k_embed
    return session

def on_image(image_path, session: Optional[MappingChatSession], cfg, prefix_mode, prefix_len, prefix_source, top_k_embed):
    session = ensure_session(session, cfg, prefix_mode, prefix_len, prefix_source, top_k_embed)
    if image_path and os.path.exists(image_path):
        session.set_image(image_path)
    return session, gr.update(value=session_history_to_pairs(session.history))

def reset_chat(session: Optional[MappingChatSession]):
    if session:
        session.reset_conversation()
    return gr.update(value=[])

def send_message(user_text: str, image_path: str, system_prompt: str,
                 session: Optional[MappingChatSession], cfg: Dict[str,Any],
                 prefix_mode, prefix_len, prefix_source, top_k_embed,
                 max_new_tokens, temperature, top_k, top_p, repetition_penalty, do_sample, use_raw_pred_seq):
    session = ensure_session(session, cfg, prefix_mode, prefix_len, prefix_source, top_k_embed)
    if system_prompt is not None:
        session.set_system_prompt(system_prompt)
    if image_path and session.image_path != image_path and os.path.exists(image_path):
        session.set_image(image_path)
    if not user_text and not cfg.get('prompt_prefix_text'):
        return session, gr.update(value=session_history_to_pairs(session.history)), ''

    # 判断是否首轮 (无 user 角色历史)
    has_user_turn = any(t['role']=='user' for t in session.history)
    prepared_user_text = user_text or ''
    if not has_user_turn and cfg.get('prompt_prefix_text'):
        # 首轮将 prefix 直接拼接在最前面，保持用户原输入在后，可选择在 UI 留空触发默认模板
        prepared_user_text = cfg['prompt_prefix_text'] + (prepared_user_text if prepared_user_text else '')
        suffix_for_turn = cfg.get('prompt_suffix_text')
    else:
        suffix_for_turn = None  # 后续轮默认不再追加 suffix，可按需修改

    try:
        reply = session.chat(
            prepared_user_text,
            suffix_text=suffix_for_turn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            use_raw_pred_seq=use_raw_pred_seq
        )
    except Exception as e:
        reply = f'生成出错: {e}'
        session.history.append({'role':'assistant','content':reply})
    pairs = session_history_to_pairs(session.history)
    return session, gr.update(value=pairs), ''

# ========== 示例初始对话 =========
SAMPLE_HISTORY: List[Tuple[str,str]] = []

# ========== 界面 =========
with gr.Blocks(title='Mapping 模型 WebUI') as demo:
    gr.Markdown('# 映射模型多轮对话 WebUI')
    gr.Markdown('基座模型为 `DeepSeek-R1-Distill-Qwen-1.5B`')

    cfg_state = gr.State(load_config())
    session_state = gr.State(None)

    chatbot = gr.Chatbot(label='对话', value=SAMPLE_HISTORY, height=480)

    with gr.Accordion('会话与系统提示', open=False):
        system_prompt = gr.Textbox(label='System Prompt', lines=3, placeholder='可选: 系统角色设定')
        with gr.Row():
            reset_btn = gr.Button('重置会话', variant='secondary')
            load_cfg_btn = gr.Button('重新加载配置')
        status = gr.Markdown('配置已加载')

    with gr.Row():
        image_input = gr.Image(label='上传图片', type='filepath')
    user_input = gr.Textbox(label='输入', lines=3, placeholder='请输入你的问题...')
    send_btn = gr.Button('发送', variant='primary')

    with gr.Accordion('生成参数', open=False):
        with gr.Row():
            max_new_tokens = gr.Slider(16, 512, value=128, step=8, label='max_new_tokens')
            temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label='temperature')
            top_k = gr.Slider(1, 100, value=40, step=1, label='top_k')
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label='top_p')
        with gr.Row():
            repetition_penalty = gr.Slider(0.8, 2.0, value=1.2, step=0.05, label='repetition_penalty')
            do_sample = gr.Checkbox(value=True, label='do_sample')
            use_raw_pred_seq = gr.Checkbox(value=False, label='use_raw_pred_seq')
        with gr.Row():
            prefix_mode = gr.Dropdown(choices=['hybrid','residual','topk_sample'], value='hybrid', label='prefix_mode')
            prefix_source = gr.Dropdown(choices=['sequence','mean'], value='sequence', label='prefix_source')
            prefix_len = gr.Slider(1, 16, value=4, step=1, label='prefix_len')
            top_k_embed = gr.Slider(1, 64, value=5, step=1, label='top_k_embed')

    gr.Examples(examples=[["请描述主要内容", None]], inputs=[user_input, image_input], label='示例')

    # 事件绑定
    load_cfg_btn.click(fn=reload_config_cb, inputs=None, outputs=[cfg_state, status])
    image_input.change(fn=on_image, inputs=[image_input, session_state, cfg_state, prefix_mode, prefix_len, prefix_source, top_k_embed], outputs=[session_state, chatbot])
    reset_btn.click(fn=reset_chat, inputs=[session_state], outputs=[chatbot])
    send_inputs = [user_input, image_input, system_prompt, session_state, cfg_state, prefix_mode, prefix_len, prefix_source, top_k_embed, max_new_tokens, temperature, top_k, top_p, repetition_penalty, do_sample, use_raw_pred_seq]
    send_outputs = [session_state, chatbot, user_input]
    send_btn.click(fn=send_message, inputs=send_inputs, outputs=send_outputs)
    user_input.submit(fn=send_message, inputs=send_inputs, outputs=send_outputs)

if __name__ == '__main__':
    demo.launch()
